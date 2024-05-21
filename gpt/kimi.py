from openai import OpenAI
 
client = OpenAI(
    api_key = "",
    base_url = "https://api.moonshot.cn/v1",
)
 
response = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[
        {"role": "user", "content": "I will give you a list of descriptions of musics. Process each individually. \
                Extract the type of the music and generate an music caption describing the genre, moods, and instruments for music tracks. \
                The music caption should be less than 25 words. Increased richness and granularity in music description. \
                Do not write introductions or explanations. Make sure you are using grammatical subject-verb-object sentences. Write an one-sentence music caption to describe it: \
                metal, heavy metal, angry music, rock music, punk rock \
                "},
    ],
    temperature=0.3,
    stream=False,
)

print(response.choices[0].message.content)
