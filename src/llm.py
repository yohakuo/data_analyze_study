import json
import logging
from typing import Any, Dict, Optional

import yaml
from openai import OpenAI

logger = logging.getLogger(__name__)


class HybridLLMService:
    def __init__(
        self,
        metrics_path: str = "metrics.yaml",
        mode: str = "local",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.mode = mode

        # Configure client
        if mode == "local":
            # Default to Ollama settings if not provided
            final_base_url = base_url or "http://localhost:11434/v1"
            final_api_key = api_key or "ollama"
            self.model = model_name or "qwen2.5:7b"
            self.client = OpenAI(base_url=final_base_url, api_key=final_api_key)
        else:
            # Online mode
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self.model = model_name or "gpt-4o"

        # Load metrics definition
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics_def = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics from {metrics_path}: {e}")
            self.metrics_def = {"metrics": []}

    def parse_intent(self, user_query: str) -> Dict[str, Any]:
        """
        Parse user natural language query into structured intent.

        Args:
            user_query: The natural language query string

        Returns:
            Dictionary containing metric_id and params
        """
        system_prompt = f"""
        你是一个专业的数据分析助手。你的任务是分析用户的自然语言查询，并将其映射到预定义的指标库中。
        
        指标库定义：
        {json.dumps(self.metrics_def, ensure_ascii=False)}
        
        请分析用户意图并提取以下信息：
        1. metric_id: 匹配的指标ID。如果不匹配，返回 null。
        2. params: 提取过滤参数，如 site_id (点位), freq (时间频率: h/D/W/M), start_date (开始日期 YYYY-MM-DD), end_date (结束日期 YYYY-MM-DD)。
        
        务必只返回合法的 JSON 对象，不包含 Markdown 格式标记（如 ```json ... ```）。格式如下：
        {{
            "metric_id": "metrics.yaml中的id 或 null",
            "params": {{
                "site_id": "提取到的点位ID 或 null",
                "freq": "h/D/W/M 或 null",
                "start_date": "YYYY-MM-DD 或 null",
                "end_date": "YYYY-MM-DD 或 null"
            }}
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")

            return json.loads(content)
        except Exception as e:
            logger.error(f"Error parsing intent: {e}")
            return {"metric_id": None, "error": str(e)}
