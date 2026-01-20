from openai import OpenAI
client=OpenAI()
def rewrite_follow_up(client,prev_query:str,follow_up:str)-> str:
    prompt =f"""You are given a previous user question and a follow-up question.
Rewrite the follow-up into a standalone question that preserves the original intent.
Previous question:
{prev_query}
Follow-up question 
{follow_up}
Output ONLY the rewritten standalone question.
Do not answer the question.
"""
    response=client.chat.completions.create (
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You rewrite follow-up questions into standalone questions."},
            {"role": "user", "content": prompt }
        ],
        temperature=0.0
        )
    return response.choices[0].message.content.strip()