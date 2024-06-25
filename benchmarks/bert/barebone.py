import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time, random, statistics
import numpy as np
from tqdm.auto import tqdm
random.seed(10)
device = "cuda:0"
model_name = "google-bert/bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float16)
model = model.eval().to(device)

phrases = [
              "In the midst of a bustling city, amidst the constant hum of traffic and the chatter of countless conversations, there exists a serene park where people come to escape the chaos. Children play on the swings, their laughter echoing through the air, while adults stroll along the winding paths, lost in thought. The trees, tall and majestic, provide a canopy of shade, and the flowers bloom in a riot of colors, adding to the park's charm. It's a place where time seems to slow down, offering a moment of peace and reflection in an otherwise hectic world.",

              "As the sun sets over the horizon, painting the sky in hues of orange, pink, and purple, a sense of calm descends over the landscape. The day has been long and filled with activity, but now, in this magical hour, everything feels different. The birds return to their nests, their evening songs a lullaby to the world. The gentle breeze carries the scent of blooming jasmine, and the stars begin to twinkle in the darkening sky. It's a time for quiet contemplation, for appreciating the beauty of nature, and for feeling a deep connection to the universe.",

              "On a remote island, far away from the noise and pollution of modern life, there is a hidden cove where crystal-clear waters lap gently against the shore. The beach, covered in soft, white sand, is a paradise for those seeking solitude and tranquility. Palm trees sway in the breeze, their fronds rustling softly, while the sun casts a warm, golden glow over everything. Here, one can forget the worries of the world and simply exist in the moment, surrounded by the natural beauty of the island and the soothing sounds of the ocean.",

              "In an ancient forest, where the trees have stood for centuries, there is a sense of timelessness that envelops everything. The air is cool and crisp, filled with the earthy scent of moss and fallen leaves. Sunlight filters through the dense canopy, creating dappled patterns on the forest floor. Birds call to one another, and small animals scurry through the underbrush. It's a place where one can feel the weight of history, where the presence of the past is almost palpable. Walking through this forest is like stepping back in time, to a world untouched by human hands.",

              "At the edge of a vast desert, where the dunes stretch out as far as the eye can see, there is a small oasis that offers a respite from the harsh conditions. A cluster of palm trees provides shade, and a clear, cool spring bubbles up from the ground, a source of life in an otherwise barren landscape. Travelers who come across this oasis are greeted with the sight of lush greenery and the sound of birdsong. It's a place of refuge and renewal, where one can rest and recharge before continuing on their journey through the endless sands.",

              "High in the mountains, where the air is thin and the landscape is rugged, there is a hidden valley that remains largely untouched by human activity. The valley is a haven for wildlife, with streams that flow with clear, cold water and meadows filled with wildflowers. The surrounding peaks, covered in snow even in the summer, stand as silent sentinels. It's a place where one can feel a profound sense of solitude and connection to nature. The beauty of the valley, with its pristine environment and abundant life, is a reminder of the importance of preserving wild places.",

              "On a quiet country road, far from the bustling cities and noisy highways, there is a small farmhouse surrounded by fields of golden wheat. The farmhouse, with its weathered wooden walls and cozy interior, is a place of warmth and hospitality. The fields, swaying gently in the breeze, are a testament to the hard work and dedication of the farmers who tend them. In the evenings, the sky is filled with stars, and the only sounds are the chirping of crickets and the distant hoot of an owl. It's a place where one can find peace and simplicity.",

              "In a quaint village, nestled in the rolling hills of the countryside, life moves at a slower pace. The cobblestone streets are lined with charming cottages, each with its own garden bursting with flowers. The village square is the heart of the community, where residents gather to catch up on news and enjoy each other's company. There's a timeless quality to the village, where traditions are upheld, and everyone knows their neighbors. It's a place where one can experience the joys of small-town living, with its close-knit community and strong sense of belonging.",

              "By the side of a tranquil lake, surrounded by dense forests and towering mountains, there is a small cabin that offers a perfect retreat from the hustle and bustle of everyday life. The cabin, with its rustic charm and cozy interior, is a place to unwind and relax. The lake, calm and mirror-like, reflects the beauty of the surrounding landscape, creating a sense of peace and serenity. It's a place where one can reconnect with nature, spend quiet moments fishing or kayaking, and enjoy the simple pleasures of life in a beautiful, natural setting.",

              "In the heart of a bustling city, there is a hidden garden that provides a peaceful escape from the noise and activity of urban life. The garden, filled with a variety of plants and flowers, is a haven for birds and butterflies. A small fountain in the center adds to the tranquil atmosphere, its gentle sound masking the distant hum of traffic. Benches are scattered throughout, offering places to sit and reflect. It's a place where city dwellers can find a moment of calm and connect with nature, even in the midst of a busy metropolis."
          ][:8]


def run_model(inputs: list[str]):
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    inputs = inputs.to(device)
    logits = model(**inputs).logits
    torch.cuda.synchronize()
    predicted_class_ids = logits.argmax(1).tolist()
    return predicted_class_ids

import random

def create_random_batch(size: int):
    result = []
    for i in range(size):
        result.append(random.choice(phrases))

    return result

def run_benchmark(batch_size:int):
    batch = create_random_batch(1)
    run_model(batch)

    times = []
    for i in range(20):
        batch = create_random_batch(batch_size)
        t0 = time.time()
        y = run_model(batch)
        t1 = time.time()
        times.append(t1 - t0)

    times = times[1:]
    average_time = sum(times) / len(times)
    median_time = statistics.median(times)
    print("Benchmark for batch size", batch_size)
    print(f"Average response time: {average_time * 1000:.2f} ms")
    print(f"Median response time: {median_time * 1000:.2f} ms")
    print("-"*20)


if __name__ == '__main__':
    for batch_size in [1, 8, 16, 32]:
        run_benchmark(batch_size)
