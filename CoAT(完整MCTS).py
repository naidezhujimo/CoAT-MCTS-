from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import re
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random

# 解析函数
def parse_solution(response):
    # 宽松匹配所有x=和y=的数值，取最后一次出现的结果
    x_values = re.findall(r'x\s*[=＝]\s*(-?\d+\.?\d*)', response)
    y_values = re.findall(r'y\s*[=＝]\s*(-?\d+\.?\d*)', response)
    
    if not x_values or not y_values:
        return None, None
    
    try:
        x = float(x_values[-1])
        y = float(y_values[-1])
        return x, y
    except:
        return None, None



tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    pad_token="<|endoftext|>"  # 显式设置pad token
)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device_map="auto",
)

# 确保pad_token有效，若不存在则使用eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
device='cuda'
model.to(device)
#============================= 实现 CoAT ==================================
class CoATNode:
    def __init__(self, parent=None, prompt='', context='', depth=0):
        self.parent = parent
        self.children = []  # 子节点列表
        self.visit_count = 0  # 访问次数
        self.total_value = 0  # 累计价值
        self.prompt = prompt  # 初始提示词
        self.context = context  # 当前节点生成内容
        self.associative_memory = []  # 关联记忆
        self.depth = depth  # 当前节点深度
        self.is_expanded = False  # 是否已扩展
    
    def uct_value(self, exploration_weight=1.414):
        """计算UCT值"""
        if self.visit_count == 0:
            return float('inf') # 未访问节点优先搜索
        exploitation = self.total_value / self.visit_count
        exploration = exploration_weight * np.sqrt(np.log(self.parent.visit_count) / (self.visit_count + 1e-6))
        return exploitation + exploration
        
    def best_child(self):
        """选择UCT值最高的子节点"""
        return max(self.children, key=lambda x: x.uct_value())
    
# CoAT MCTS
class CoATFramework:
    def __init__(self, model, tokenizer, max_iter=100, max_depth=5, num_simulations=50):
        self.model = model
        self.tokenizer = tokenizer
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.external_brain = self.init_external_brain()
    
    def init_external_brain(self):
        """初始化外部数学库"""
        return {
            "消元法": "联立方程消去变量：方程1 + 方程2 → 3x = 9 → x=3",
            "代入法": "从方程1解出y=8-x，代入方程2得 2x - (8-x) =1 → x=3",
            "验证步骤": "将x=3代入原方程验证：3 + y=8 → y=5"
        }
    
    def retrieve_associative_memory(self, context):
        """动态关联记忆检索"""
        keywords = ["消元", "代入", "验证", "解"]
        for kw in keywords:
            if kw in context:
                return self.external_brain.get(kw, "")
        return ""
    
    def evaluate_node(self, node):
        """节点价值评估"""
        # 生成质量
        full_text = node.prompt + node.context
        ppl = compute_perplexity(self.model, self.tokenizer, full_text)
        gen_score = 1 / (ppl + 1e-6)

        # 关联内容质量
        am_score = 0.2 * len(node.associative_memory) # 每条关联内容+0.2分
        return gen_score + am_score
    
    def expand_node(self, node):
        """节点扩展, 生成候选内容并关联记忆"""
        if node.is_expanded:
            return
        # 生成候选内容（基于当前上下文）
        input_text = node.prompt + node.context
        inputs = self.tokenizer(input_text, return_tensors="pt").to(device)
        inputs["attention_mask"] = inputs.input_ids.ne(tokenizer.pad_token_id).int()
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=100,
            num_return_sequences=3, # 每个节点生成3个候选
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        # 创建子节点
        for seq in outputs:
            child_text = self.tokenizer.decode(seq, skip_special_tokens=True)
            child = CoATNode(
                parent=node,
                prompt=node.prompt,
                context=child_text[len(input_text):], # 仅保留新增内容
                depth=node.depth+1
            )
            # 关联记忆搜索
            child.associative_memory = self.retrieve_associative_memory(child.context)
            node.children.append(child)
        node.is_expanded = True

    def simulate(self, node):
        """蒙特卡洛模拟: 随机游走到叶节点并评估"""
        current_depth = node.depth
        while current_depth < self.max_depth:
            if not node.children:
                self.expand_node(node)
            if not node.children:
                break # 无子节点可扩展
            node = random.choice(node.children) # 随机选择子节点
            current_depth += 1
        return self.evaluate_node(node)
    
    def backpropagate(self, node, value):
        """回溯更新节点价值"""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent
        
    def mcts_search(self, root):
        """蒙特卡洛树搜索"""
        for _ in range(self.max_iter):
            # 选择阶段
            node = root
            while node.children:
                node = node.best_child()
            
            # 扩展阶段
            if node.depth < self.max_depth and not node.is_expanded:
                self.expand_node(node)

            # 模拟阶段
            total_sim_value = 0
            for _ in range(self.num_simulations):
                sim_value = self.simulate(node)
                total_sim_value += sim_value
            avg_sim_value = total_sim_value / self.num_simulations

            # 回溯更新
            self.backpropagate(node, avg_sim_value)

        # 选择最优路径
        best_node = root.best_child()
        return best_node
    
# CoAT生成函数
def generate_with_coat(prompt, coat):
    # 初始化搜索树
    root = CoATNode(prompt=prompt, context="")
    coat.expand_node(root) # 初始扩展

    # MCTS搜索
    best_node = coat.mcts_search(root)

    # 构建最终响应
    full_response = best_node.context
    if best_node.associative_memory:
        full_response += f"\n[关联知识] {best_node.associative_memory}"
    
    # 回溯生成完整路径
    path = []
    current_node = best_node
    while current_node.parent:
        path.append(current_node.context)
        current_node = current_node.parent
    path.reverse()
    full_response = "\n".join(path) + "\n" + full_response
    
    return full_response


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


# 设置生成参数
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
num_candidates = 5  # 每轮生成的候选响应数量
ground_truth = (3, 5) # 方程组的真实解

# 奖励函数：根据解的正确性得分
def reward_function(parsed_solution):
    if parsed_solution is None:
        return -1.0 # 无效解惩罚
    x_pred, y_pred = parsed_solution
    # 计算误差并归一化 [0,1]
    max_error = 8  # 最大可能误差（如x=8,y=0时误差为5+8=13，但需根据问题调整）
    error = (abs(x_pred - ground_truth[0]) + abs(y_pred - ground_truth[1])) / max_error
    # 思路切换惩罚
    switch_count = sum([1 for token in switch_tokens if token in response])
    penalty = 0.05 * switch_count # 每次切换扣 0.05 分
    return max(0.0, 1.0 - error - penalty)

def tpo_optimization(initial_prompt):
    cache = [] # 存储（响应、奖励分）的缓存

    coat = CoATFramework(model, tokenizer)
    # 初始生成候选
    candidates = [generate_with_coat(initial_prompt, coat) for _ in range(num_candidates)]
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

print("============ TPO模型 =============")
# 运行TPO优化
optimized_response, feedback = tpo_optimization(cot_prompt)
ppl = compute_perplexity(model, tokenizer, optimized_response)
print(f"输入：{feedback}...\n输出：{optimized_response}...\n困惑度：{ppl:.2f}")

with open('tpo_response.txt','w', encoding='utf-8') as f:
    f.write(f"输入：{feedback}\n输出：{optimized_response}\n困惑度：{ppl:.2f}")