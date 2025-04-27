from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import re
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 解析函数
def parse_solution(response):
    # 宽松匹配所有x=和y=的数值，取最后一次出现的结果
    x_values = re.findall(r'x\s*[=＝]\s*(-?\d+\.?\d*([eE]-?\d+)?)', response)
    y_values = re.findall(r'y\s*[=＝]\s*(-?\d+\.?\d*([eE]-?\d+)?)', response)
    
    if not x_values or not y_values:
        return None, None
    
    try:
        x = float(x_values[-1])
        y = float(y_values[-1])
        return x, y
    except (ValueError, IndexError, TypeError):
        return None, None



tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    pad_token="<|endoftext|>"  # 显式设置pad token
)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device_map="auto",
    # torch_dtype=torch.float16
)

# 确保pad_token有效，若不存在则使用eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
device='cuda'
model.to(device)

#============================= 实现 CoAT 框架==================================
class CoATNode:
    def __init__(self, parent=None, prompt=None, context=""):
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.value = 0.0
        self.context = context # 当前节点生成内容
        self.associative_memory = [] # 关联记忆内容
        self.prompt = prompt
        self.step_id = 0 # 步骤编号

class CoATFramework:
    def __init__(self, model, tokenizer, max_depth=3, num_candidates=3):
        self.model = model
        self.tokenizer = tokenizer
        self.max_depth = max_depth
        self.num_candidates = num_candidates
        self.external_brain = None  # 外部知识库接口

    def retrieve_associative_memory(self, context):
        """外部知识检索"""
        formulas = {
            "消元法": "通过联立方程消去一个变量，例如：方程1 + 方程2 → 3x=9",
            "代入法": "从方程1解出y=8-x，代入方程2 → 2x - (8-x) =1",
            "符号错误": "注意负号处理：例如 2x - (8-x) =1 → 2x -8 +x =1 (非2x-8-x=1)"
        }
        keywords = ['消元', '代入', '联立', '错误', '验证']
        for kw in keywords:
            if kw in context:
                return formulas.get(kw, "")
        return ""
    
    def evaluate_node(self, node):
        """评估节点价值：生成质量 + 关联内容质量"""
        # 生成质量评分
        ppl = compute_perplexity(self.model, self.tokenizer, node.context)
        gen_score = 1 / (ppl + 1e-6)

        # 关联内容质量评分
        am_score = 1.0 if any(kw in node.associative_memory for kw in ["代入法", "消元法"]) else 0.5
        return gen_score + 0.8 * am_score
    
    def expand_node(self, node):
        """扩展节点，生成候选内容并关联记忆"""
        inputs = self.tokenizer(node.prompt + node.context, return_tensors='pt').to(device)
        inputs["attention_mask"] = inputs.input_ids.ne(tokenizer.pad_token_id).int()
        outputs = self.model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=100,
        num_return_sequences=self.num_candidates,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        )
        candidates = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

        # 创建子节点
        for cand in candidates:
            child = CoATNode(parent=node, prompt=node.prompt, context=cand)
            # 关联记忆搜索
            child.associative_memory = self.retrieve_associative_memory(cand)
            node.children.append(child)

    def mcts_search(self, root):
        """简化版蒙特卡洛树搜索"""
        for _ in range(self.max_depth):
            node = root
            # 选择阶段：选择价值最高的子节点
            while node.children:
                node = max(node.children, key=lambda x: x.value + np.sqrt(np.log(x.parent.visit_count)/(x.visit_count+1)))

            # 扩展截断
            if not node.children:
                self.expand_node(node)

            # 模拟阶段：评估叶子节点
            value = self.evaluate_node(node)

            # 回溯更新
            while node:
                node.visit_count += 1
                node.value += (value - node.value) / node.visit_count
                node = node.parent
        return max(root.children, key=lambda x: x.value)
    

#============================= 实现 TIP Logits 处理器==================================
# 定义思路切换相关的触发词
switch_tokens = [
    '另一种方法', 'alternatively', '或者', '换一种思路',
    '但是', '另一方面', '然而'
]

# 通过分词器转换为 token id
switch_tokens_ids = tokenizer.convert_tokens_to_ids(switch_tokens)

from transformers import LogitsProcessor

class TIPLogitsProcessor(LogitsProcessor):
    def __init__(self, switch_token_ids, alpha=3.0, beta=300):
        self.switch_token_ids = switch_token_ids
        self.alpha = alpha # 惩罚强度
        self.beta = beta # 惩罚时间
        self.current_thought_start = 0 # 当前思路的起始位置

    def __call__(self, input_ids, scores):
        # 检查是否触发新思路
        last_token = input_ids[0][-1].item()
        if last_token in self.switch_token_ids:
            self.current_thought_start = input_ids.shape[-1] # 记录新思路的起始位置
        
        # 计算当前处理 token 是否在惩罚窗口中
        current_position = input_ids.shape[-1]
        if current_position < self.current_thought_start + self.beta:
            # 对切换 token 施加惩罚
            for token_id in self.switch_token_ids:
                scores[:, token_id] -= self.alpha
        
        return scores


# CoT
def generate_with_cot(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    inputs["attention_mask"] = inputs.input_ids.ne(tokenizer.pad_token_id).int()

    # 添加 TIP Logits 处理器
    logits_processor = [TIPLogitsProcessor(switch_tokens_ids, alpha=3.0, beta=300)]

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=512,
        temperature=0.85, 
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        logits_processor=logits_processor # 注入TIP处理器
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

problem = "解方程组：\n方程1: x + y = 8\n方程2: 2x - y = 1"

# 修改后的CoT提示（增加改进空间）
cot_prompt = f"""
请逐步解决以下问题：

{problem}

分步推理要求：
1. 明确标注步骤编号
2. 展示完整的代数运算过程
3. 最终解用方框标出（如：x=3, y=5）
"""

# CoAT
def generate_hybrid(prompt, coat, max_steps=3):
    """结合CoT步骤约束和CoAT多路径搜索"""
    root = CoATNode(prompt=prompt, context="")
    current_step = 0
    
    while current_step < max_steps:
        # 扩展节点前清空旧子节点（避免历史干扰）
        root.children = []
        
        # CoAT扩展（生成当前步骤的多个候选）
        coat.expand_node(root)
        
        # ==== 调试输出：检查候选内容 ====
        print(f"步骤{current_step+1}候选内容：")
        for i, child in enumerate(root.children):
            print(f"候选{i+1}: {child.context[:50]}...")  # 打印前50字符
        
        # 选择最优子节点（需满足CoT步骤约束）
        best_child = None
        for child in root.children:
            # ==== 改进匹配逻辑：使用正则表达式 ====
            if re.search(fr'步骤{current_step+1}[\s:：]', child.context):
                if best_child is None or child.value > best_child.value:
                    best_child = child
        
        if not best_child:
            print(f"步骤{current_step+1}无合规候选，提前终止")
            break
        
        # 更新上下文和步骤
        root = best_child
        current_step += 1
    
    # 拼接最终响应（添加空值保护）
    final_response = "\n".join([node.context for node in get_path(root) if node.context.strip()])
    
    # ==== 容错处理：确保响应非空 ====
    if not final_response.strip():
        final_response = "模型未能生成有效解答。请检查输入或调整参数。"
        print("警告：生成响应为空，已替换为默认提示")
    
    return final_response

def get_path(node):
    """回溯获取完整推理路径（修复逆序逻辑）"""
    path = []
    current_node = node
    while current_node:
        path.append(current_node)
        current_node = current_node.parent
    return reversed(path)  # 从根节点到叶节点的正确顺序

# 困惑度计算函数
def compute_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    tokens = inputs["input_ids"]
    nll = F.nll_loss(log_probs[:, :-1].contiguous().view(-1, log_probs.size(-1)),
                     tokens[:, 1:].contiguous().view(-1),
                     reduction='mean')
    return torch.exp(nll).item()


response = generate_with_cot(cot_prompt)
ppl = compute_perplexity(model, tokenizer, cot_prompt)
print("============ 原始模型 =============")
print(f"输入：{cot_prompt}...\n输出：{response}...\n困惑度：{ppl:.2f}")

with open('pre_response.txt','w', encoding='utf-8') as f:
    f.write(f"输入：{cot_prompt}\n输出：{response}\n困惑度：{ppl:.2f}")


# ============================ TPO =========================
num_iterations = 3  # TPO迭代次数
num_candidates = 3  # 每轮生成的候选响应数量
ground_truth = (3, 5) # 方程组的真实解

# 奖励函数：根据解的正确性得分
def reward_function(parsed_solution):
    if parsed_solution is None or None in parsed_solution:
        return -1.0 # 无效解惩罚
    x_pred, y_pred = parsed_solution
    # 计算误差并归一化 [0,1]
    max_error = 8  # 最大可能误差（如x=8,y=0时误差为5+8=13，但需根据问题调整）
    error = (abs(x_pred - ground_truth[0]) + abs(y_pred - ground_truth[1])) / max_error
    # 思路切换惩罚
    switch_count = sum([1 for token in switch_tokens if token in response])
    penalty = 0.05 * switch_count # 每次切换扣 0.05 分

    am_score = 0.3 if "关联知识：" in response else 0
    return max(0.0, 1.0 - error - penalty + am_score)

def tpo_optimization(initial_prompt):
    cache = [] # 存储（响应、奖励分）的缓存
    coat = CoATFramework(model, tokenizer) # 初始化CoAT框架

    # 初始生成候选
    candidates = [generate_hybrid(initial_prompt, coat) for _ in range(num_candidates)]
    for resp in candidates:
        x, y = parse_solution(resp)
        score = reward_function((x, y))
        cache.append((resp, score))

    # TPO迭代
    for _ in tqdm(range(num_iterations), desc='TPO Train'):
        # 选择最优和最差响应
        best_resp = max(cache, key=lambda x:x[1])[0]
        worst_resp = min(cache, key=lambda x:x[1])[0]

        # 生成文本反馈（改进建议）
        feedback_prompt = f"""
        以下是两个解方程组的示例：

        **优秀示例**：
        {best_resp}

        **较差示例**：
        {worst_resp}

        请分析优秀示例的优点和较差示例的不足，并提出改进建议：
        1. 步骤完整性（是否遗漏验证步骤）
        2. 计算准确性（是否存在算术错误）
        3. 表达清晰度（是否使用明确标记）

        **强制修正要求**：
        - 必须验证消元步骤：3x=9 → x=3
        - 若出现矛盾结论，必须重新计算
        """
        feedback_prompt += """
        **注意**：反馈需满足以下要求：
        - 分析必须具体，避免复述解题过程
        - 改进建议不超过3条
        - 强制使用LaTeX公式标注关键步骤
        """

        # 在反馈生成中强制加入关联记忆验证
        feedback_prompt += """
        **新增关联性检查要求**：
        - 必须引用至少一个数学方法（如消元法/代入法）
        - 若发现步骤跳跃，需补充中间计算
        """
        feedback = generate_with_cot(feedback_prompt)

        # 基于反馈生成新候选
        new_candidates = [generate_with_cot(f"{initial_prompt}\n改进建议：{feedback}") 
                         for _ in range(num_candidates)]
        
        # 更新缓存
        for resp in new_candidates:
            x, y = parse_solution(resp)
            score = reward_function((x, y))
            cache.append((resp, score))

    # 返回最高分响应
    return max(cache, key=lambda x:x[1])[0], feedback

# ========== 运行混合模型 ==========
coat = CoATFramework(model, tokenizer)
hybrid_response = generate_hybrid(cot_prompt, coat)
hybrid_ppl = compute_perplexity(model, tokenizer, hybrid_response) 
print("============ CoAT+CoT混合模型 =============")
print(f"输出：{hybrid_response}\n困惑度：{hybrid_ppl:.2f}")

# ========== 更新TPO调用 ==========
optimized_response, feedback = tpo_optimization(cot_prompt)
dpo_ppl = compute_perplexity(model, tokenizer, optimized_response)
with open('tpo_response.txt','w', encoding='utf-8') as f:
    f.write(f"输入：{feedback}\n输出：{hybrid_response}\n困惑度：{dpo_ppl:.2f}")
