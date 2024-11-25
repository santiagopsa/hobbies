import openai

openai.api_key = "sk-f3rQTzIsECXmppO8PivLFyi7A0ZP4MRT1c-Fp8N0RpT3BlbkFJMHLP9okjWn24d7s_psSHxdebNpib1KxqcnVeLpNJIA"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # Replace with "gpt-4" if needed
    messages=[
        {"role": "system", "content": "You are a trading expert."},
        {"role": "user", "content": "Should I buy, sell, or hold BTC given these trends: [Trend data here]?"}
    ],
    max_tokens=100
)

print(response['choices'][0]['message']['content'])
