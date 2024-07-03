from openai import OpenAI
import re, sys
import pandas as pd
from tqdm import tqdm


client = OpenAI(
    api_key=YOUR_API_KEY_HERE
)


def generate_prompt(gt, candidate):
    prompt = f"You are a grader of soccer game commentaries. There is a predicted commentary by AI model about a soccer game video clip and you need to score it comparing with ground truth. \n\nYou should rate an integer score from 0 to 10 about the degree of similarity with ground truth commentary (The higher the score, the more correct the candidate is). You must first consider the accuracy of the soccer events, then to consider about the semantic information in expressions and the professional soccer terminologies. The names of players and teams are masked by \"[PLAYER]\" and \"[TEAM]\". \n\nThe ground truth commentary of this soccer game video clip is:\n\n\"{gt}\"\n\n I need you to rate the following predicted commentary from 0 to 10:\n\n\"{candidate}\"\n\nThe score you give is (Just return one number, no other word or sentences):"
    return prompt

def score(client, prompt):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are an expert in professional soccer commentary."},
        {"role": "user", "content": prompt},
    ],
    stop=["\n"],
    temperature=0,
    max_tokens=5,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0

    )

    res = completion.choices[0].message.content
    result = re.search(r'\b([0-9]|10)\b', res).group(0)
    return int(result)

file_path = sys.argv[1] 
data = pd.read_csv(file_path)
if 'llm_score' not in data.columns:
    data['llm_score'] = None
    data.to_csv(file_path, index=False)

for start in tqdm(range(0, pd.read_csv(file_path).shape[0], 1)):
    data = pd.read_csv(file_path)

    end = start + 10
    group_data = data.iloc[start:end]

    if 'llm_score' not in data.columns:
        data['llm_score'] = None

    for index, row in group_data.iterrows():
        if pd.isna(row['llm_score']): 
            gt = row['anoymized']
            candidate = row['predicted_res']
            prompt = generate_prompt(gt, candidate)
            result = score(client, prompt) 
            data.at[index, 'llm_score'] = result

    data.to_csv(file_path, index=False)
