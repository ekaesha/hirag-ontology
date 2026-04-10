import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

JUDGE_PROMPT = """You are evaluating the quality of an answer to a medical question.

Question: {question}

Answer to evaluate:
{answer}

Rate the answer on these 4 criteria, each from 0 to 10:

1. Comprehensiveness (0-10): Does the answer cover all relevant aspects?
   0 = completely missing key info, 10 = fully comprehensive

2. Empowerment (0-10): Does the answer give actionable, useful information?
   0 = vague/useless, 10 = highly actionable

3. Diversity (0-10): Does the answer present varied relevant information?
   0 = repetitive/one-sided, 10 = rich and varied

4. Overall (0-10): Overall quality as a medical reference.
   0 = wrong/harmful, 10 = excellent

Return ONLY valid JSON:
{{
  "comprehensiveness": <0-10>,
  "empowerment": <0-10>,
  "diversity": <0-10>,
  "overall": <0-10>,
  "comment": "<one sentence about main strength or weakness>"
}}"""


class LLMJudge:
    def __init__(self, model="deepseek-chat"):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.model = model

    def evaluate(self, question, answer):
        prompt = JUDGE_PROMPT.format(
            question=question,
            answer=answer,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            scores = json.loads(response.choices[0].message.content)
            for field in ["comprehensiveness", "empowerment",
                          "diversity", "overall"]:
                if field not in scores:
                    scores[field] = 5.0
            return scores
        except Exception as e:
            print(f"  [Judge] Error: {e}")
            return {
                "comprehensiveness": 0, "empowerment": 0,
                "diversity": 0, "overall": 0,
                "comment": f"Error: {e}",
            }