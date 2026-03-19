import random
import json
import os
from PIL import Image, ImageDraw, ImageFont
import sympy as sp
from tqdm import tqdm

# --------------------------------------------------
# create training data: each image contains some math problems
# --------------------------------------------------
OUTPUT_DIR = "../data"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
NUM_WORKSHEETS = 8
PROBLEMS_PER_PAGE = 5

IMAGE_SIZE = (1240, 1754)  # A4 @ 150dpi
MARGIN_X = 80
MARGIN_Y = 100
LINE_SPACING = 90

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_SIZE = 42

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

# --------------------------------------------------
# Problem generators
# --------------------------------------------------
def arithmetic_problem():
    a, b = random.randint(1, 20), random.randint(1, 20)
    op = random.choice(["+", "-", "*", "/"])

    if op == "/":
        a = a * b  # ensure integer result

    expr = f"{a} {op} {b}"
    answer = eval(expr)

    return expr, str(answer)

def linear_equation():
    x = sp.symbols("x")
    a = random.randint(1, 10)
    b = random.randint(1, 20)
    c = random.randint(1, 20)

    expr = f"{a}x + {b} = {c}"
    solution = sp.solve(a * x + b - c, x)[0]

    return expr, f"x = {solution}"

def fraction_problem():
    a, b = random.randint(1, 9), random.randint(2, 9)
    c, d = random.randint(1, 9), random.randint(2, 9)

    expr = f"{a}/{b} + {c}/{d}"
    result = sp.Rational(a, b) + sp.Rational(c, d)

    return expr, str(result)

def generate_problem():
    choice = random.choice([
        arithmetic_problem,
        linear_equation,
        fraction_problem
    ])
    return choice()

# --------------------------------------------------
# Worksheet rendering
# --------------------------------------------------
def render_worksheet(problems, filename):
    img = Image.new("RGB", IMAGE_SIZE, "white")
    draw = ImageDraw.Draw(img)

    y = MARGIN_Y
    draw.text((MARGIN_X, y - 60), "Math Worksheet", fill="black", font=font)

    for i, (expr, _) in enumerate(problems, start=1):
        draw.text(
            (MARGIN_X, y),
            f"{i}. {expr}",
            fill="black",
            font=font
        )
        y += LINE_SPACING

    img.save(filename)

# --------------------------------------------------
# Main generation loop
# --------------------------------------------------
records = []

for idx in tqdm(range(NUM_WORKSHEETS)):
    problems = [generate_problem() for _ in range(PROBLEMS_PER_PAGE)]

    image_name = f"worksheet_{idx:06d}.png"
    image_path = os.path.join(IMAGE_DIR, image_name)

    render_worksheet(problems, image_path)

    answer_text = "\n".join(
        f"{i+1}) {expr} → {ans}"
        for i, (expr, ans) in enumerate(problems)
    )

    records.append({
        "image": image_name,
        "question": "Solve the following problems.",
        "answer": answer_text
    })

# --------------------------------------------------
# Save dataset
# --------------------------------------------------
with open(os.path.join(OUTPUT_DIR, "train.json"), "w") as f:
    json.dump(records, f, indent=2)

print("Dataset generation complete.")