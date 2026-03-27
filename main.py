import json
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple

class IntentClassifier:
    """意图分类器基类"""
    def predict(self, query: str) -> str:
        raise NotImplementedError

class RuleBasedClassifier(IntentClassifier):
    """基于规则的分类器（旧版本）"""
    def __init__(self):
        self.rules = {
            "产品咨询": ["怎么用", "功能", "介绍", "多少钱"],
            "技术支持": ["错误", "问题", "无法", "bug", "崩溃"],
            "账户管理": ["登录", "注册", "密码", "账号"],
            "其他": []
        }
    
    def predict(self, query: str) -> str:
        """基于关键词匹配的意图分类"""
        query_lower = query.lower()
        for intent, keywords in self.rules.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return intent
        return "其他"

class LLMBasedClassifier(IntentClassifier):
    """基于大模型的分类器（优化版本）"""
    def __init__(self, mock_api: bool = True):
        self.mock_api = mock_api
        self.intent_map = {
            "product_inquiry": "产品咨询",
            "technical_support": "技术支持", 
            "account_management": "账户管理",
            "billing_issue": "计费问题",
            "feature_request": "功能建议",
            "other": "其他"
        }
    
    def predict(self, query: str) -> str:
        """模拟大模型API调用进行意图分类"""
        if self.mock_api:
            # 模拟API调用延迟
            time.sleep(0.05)
            
            # 模拟大模型更细粒度的分类
            query_lower = query.lower()
            
            if any(word in query_lower for word in ["价格", "费用", "付费", "续费"]):
                return self.intent_map["billing_issue"]
            elif any(word in query_lower for word in ["建议", "希望", "增加", "添加"]):
                return self.intent_map["feature_request"]
            elif any(word in query_lower for word in ["登录", "密码", "账号", "注册"]):
                return self.intent_map["account_management"]
            elif any(word in query_lower for word in ["错误", "bug", "崩溃", "问题"]):
                return self.intent_map["technical_support"]
            elif any(word in query_lower for word in ["功能", "怎么用", "介绍", "教程"]):
                return self.intent_map["product_inquiry"]
            else:
                return self.intent_map["other"]
        else:
            # 实际项目中这里会调用真实的大模型API
            # response = requests.post(api_url, json={"query": query})
            # return response.json()["intent"]
            return self.intent_map["other"]

class ABTestManager:
    """A/B测试管理器"""
    def __init__(self, group_a_ratio: float = 0.5):
        self.group_a_ratio = group_a_ratio
        self.classifier_a = RuleBasedClassifier()  # 对照组
        self.classifier_b = LLMBasedClassifier()   # 实验组
        self.results = {
            "group_a": {"total": 0, "correct": 0},
            "group_b": {"total": 0, "correct": 0}
        }
    
    def assign_group(self, user_id: str) -> str:
        """分配用户到A/B组"""
        hash_value = hash(user_id) % 100
        return "group_a" if hash_value < self.group_a_ratio * 100 else "group_b"
    
    def predict_intent(self, user_id: str, query: str, true_intent: str = None) -> Tuple[str, str]:
        """预测意图并记录结果"""
        group = self.assign_group(user_id)
        
        if group == "group_a":
            predicted = self.classifier_a.predict(query)
            classifier_name = "rule_based"
        else:
            predicted = self.classifier_b.predict(query)
            classifier_name = "llm_based"
        
        # 如果有真实标签，记录准确率
        if true_intent:
            self.results[group]["total"] += 1
            if predicted == true_intent:
                self.results[group]["correct"] += 1
        
        return predicted, classifier_name
    
    def get_metrics(self) -> Dict:
        """获取测试指标"""
        metrics = {}
        for group in ["group_a", "group_b"]:
            total = self.results[group]["total"]
            correct = self.results[group]["correct"]
            metrics[group] = {
                "total_queries": total,
                "correct_predictions": correct,
                "accuracy": correct / total if total > 0 else 0
            }
        return metrics

def load_test_data() -> List[Tuple[str, str]]:
    """加载测试数据"""
    return [
        ("怎么登录账号？", "账户管理"),
        ("系统出现错误代码500", "技术支持"),
        ("这个产品有什么功能？", "产品咨询"),
        ("续费价格是多少？", "计费问题"),
        ("希望能增加导出功能", "功能建议"),
        ("今天天气怎么样", "其他")
    ]

def main():
    """主函数：模拟A/B测试流程"""
    print("=== 智能知识库问答组件优化 A/B测试 ===")
    print("开始时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    # 初始化A/B测试管理器
    ab_test = ABTestManager(group_a_ratio=0.5)
    test_data = load_test_data()
    
    print("测试数据样本:")
    for query, intent in test_data:
        print(f"  - 问题: {query} | 真实意图: {intent}")
    print()
    
    # 模拟用户请求
    print("模拟用户请求处理...")
    for i, (query, true_intent) in enumerate(test_data * 10):  # 重复10次增加样本量
        user_id = f"user_{i % 100}"  # 模拟100个不同用户
        predicted_intent, classifier = ab_test.predict_intent(user_id, query, true_intent)
        
        if i < 3:  # 打印前3个示例
            print(f"用户{user_id}: '{query}'")
            print(f"  预测: {predicted_intent} ({classifier}) | 真实: {true_intent}")
            print(f"  结果: {'✓' if predicted_intent == true_intent else '✗'}")
    
    # 输出测试结果
    print("\n=== A/B测试结果 ===")
    metrics = ab_test.get_metrics()
    
    for group, data in metrics.items():
        classifier_type = "规则分类器" if group == "group_a" else "大模型分类器"
        print(f"\n{classifier_type} ({group}):")
        print(f"  总查询数: {data['total_queries']}")
        print(f"  正确预测: {data['correct_predictions']}")
        print(f"  准确率: {data['accuracy']:.2%}")
    
    # 计算提升效果
    acc_a = metrics["group_a"]["accuracy"]
    acc_b = metrics["group_b"]["accuracy"]
    
    if acc_a > 0:
        improvement = (acc_b - acc_a) / acc_a
        print(f"\n=== 优化效果 ===")
        print(f"准确率提升: {improvement:.1%}")
        print(f"用户首次提问解决率预估提升: {min(improvement * 1.3, 0.3):.1%}")  # 模拟估算
    
    print("\n测试完成时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()