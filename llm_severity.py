from dotenv import load_dotenv
import os
from openai import OpenAI
import json

# Load environment variables from .env file
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

# Model selection
# qwen_model = "qwen-plus",
qwen_model = "qwen3-235b-a22b-thinking-2507" 
# qwen_model = "qwen3-235b-a22b-instruct-2507" 

# Test it
my_messages = [
    {"role": "system", "content": """You are analyzing robot arm collision damage severity. Think through this assessment step by step.

SCENARIO: A robot arm mechanically fails and physically strikes objects with significant force.

THINKING PROCESS:
1. First consider: What physically happens when the robot arm impacts this object?
2. Then assess: What are the immediate consequences (injury, damage, hazards)?
3. Finally apply scoring rules consistently across all objects

MANDATORY SCORING RULES:
- ANY HUMAN (adult/child/elderly) = 10 (severe injury/death risk)
- Fire/explosion/toxic release = 9-10 (life-threatening hazards)
- Sharp objects becoming projectiles = 7-8 (serious injury risk)  
- Expensive/critical equipment = 5-7 (significant cost/disruption)
- Cheap replaceable items = 3-5 (minor cost)
- Soft/flexible materials = 1-2 (negligible damage)

KEY PRINCIPLE: Human life > Property damage > Minor inconvenience

CRITICAL: Before scoring each object, ask yourself:
- "Is this a human being?" → If yes, score = 10
- "Does impact create life-threatening hazard?" → If yes, score = 9-10
- "What's the worst realistic outcome from robot impact?"

Think through your reasoning, then respond with ONLY the JSON array:
[{"object": "name", "danger_score": number, "reasoning": "robot arm impact causes..."}]"""},

    {"role": "user", "content": """A heavy industrial robot arm holding a ceramic plate suffers joint failure and will strike these objects with significant force:

Objects: ["laptop computer", "elderly person", "glass beaker with chemicals", "cardboard box", "metal desk"]

Think through each collision scenario, then score the danger/damage severity."""}
]

completion = client.chat.completions.create(
    model= qwen_model,
    messages=my_messages
)

response_text = completion.choices[0].message.content

data = json.loads(response_text)
print(json.dumps(data, indent=2))


'''
Qwen-instruct

[
  {
    "object": "laptop computer",
    "danger_score": 5,
    "reasoning": "robot arm impact causes significant damage to expensive electronic equipment, resulting in complete destruction and moderate operational disruption"
  },
  {
    "object": "elderly person",
    "danger_score": 10,
    "reasoning": "robot arm impact causes severe injury or death due to high force collision with a human being, regardless of age or health status"
  },
  {
    "object": "glass beaker with chemicals",
    "danger_score": 10,
    "reasoning": "robot arm impact shatters the beaker, releasing hazardous chemicals that may cause toxic exposure, fire, or explosion, creating life-threatening conditions"
  },
  {
    "object": "cardboard box",
    "danger_score": 3,
    "reasoning": "robot arm impact crushes the box, causing minor damage with negligible cost or safety risk as it is a cheap and replaceable item"
  },
  {
    "object": "metal desk",
    "danger_score": 6,
    "reasoning": "robot arm impact causes structural deformation or damage to durable, critical workspace equipment, resulting in significant repair costs and operational disruption"
  }
]


Qwen-thinking
[
  {
    "object": "laptop computer",
    "danger_score": 5,
    "reasoning": "robot arm impact causes destruction of moderately expensive equipment with data loss and replacement cost, but no life-threatening hazards"
  },
  {
    "object": "elderly person",
    "danger_score": 10,
    "reasoning": "robot arm impact causes severe trauma or fatal injury to human with high vulnerability due to age"
  },
  {
    "object": "glass beaker with chemicals",
    "danger_score": 9,
    "reasoning": "robot arm impact causes glass shattering and toxic chemical release creating immediate life-threatening inhalation/contact hazards"
  },
  {
    "object": "cardboard box",
    "danger_score": 1,
    "reasoning": "robot arm impact causes negligible damage to soft, flexible, and cheap disposable material"
  },
  {
    "object": "metal desk",
    "danger_score": 4,
    "reasoning": "robot arm impact causes moderate structural damage to durable but replaceable furniture with minor operational disruption"
  }
]

'''