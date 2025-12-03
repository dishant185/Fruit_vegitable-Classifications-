from flask import Flask, render_template, request, Response
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io, base64
import cv2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ============================================================
#                    LOAD MODEL
# ============================================================
MODEL_PATH = "veg_fruit_classify.keras"
model = load_model(MODEL_PATH)

IMG_HEIGHT = 180
IMG_WIDTH = 180

# ============================================================
#                MODEL LABELS (YOUR CATEGORY LIST)
# ============================================================
data_cat = [
 'apple','avocado','banana','barbados cherry','beetroot','bell pepper',
 'berries','blackberry','brocolli','cabbage','cantaloupe','capsicum',
 'carrot','cauliflower','cherry','chilli pepper','corn','courgette',
 'cucumber','dates','dragon fruit','eggplant','fig','garlic','ginger',
 'grapes','jalepeno','kiwi','lemon','lettuce','lychee','mango','nectarine',
 'olive','onion','orange','paprika','passion','pawpaw','peach','pear','peas',
 'pepino','pineapple','plum','pomegranate','potato','pumpkin','raddish',
 'soy beans','spinach','strawberry','sugar apple','sweetcorn','sweetpotato',
 'tangarine','tomato','turnip','watermelon'
]

# ============================================================
#            BASIC NUTRITION & RECIPE DATA (SHORT)
#  (If you want full 60-item detailed nutrition list, tell me)
# ============================================================
NUTRITION_RECIPES = {
    "apple": {
        "nutrition": {"Calories": "52 kcal", "Fiber": "2.4 g", "Vitamin C": "7% DV"},
        "recipes": [
            {"title": "Apple Oats", "steps": "Mix chopped apples with oats and milk."},
            {"title": "Apple Salad", "steps": "Combine apple cubes, honey, and lemon."}
        ]
    },
    "avocado": {
        "nutrition": {"Calories": "160 kcal", "Healthy Fats": "15 g", "Fiber": "7 g"},
        "recipes": [
            {"title": "Avocado Toast", "steps": "Spread mashed avocado on toast."},
            {"title": "Quick Guacamole", "steps": "Mash avocado with salt and lime."}
        ]
    },
    "banana": {
        "nutrition": {"Calories": "89 kcal", "Potassium": "358 mg", "Carbs": "23 g"},
        "recipes": [
            {"title": "Banana Smoothie", "steps": "Blend banana with milk."},
            {"title": "Banana Pancake", "steps": "Mix mashed banana with egg and fry."}
        ]
    },
    "barbados cherry": {
        "nutrition": {"Vitamin C": "Very High", "Calories": "32 kcal"},
        "recipes": [
            {"title": "Cherry Juice", "steps": "Blend cherries with water."},
            {"title": "Cherry Mix", "steps": "Add cherries to fruit bowl."}
        ]
    },
    "beetroot": {
        "nutrition": {"Calories": "43 kcal", "Iron": "8% DV"},
        "recipes": [
            {"title": "Beet Salad", "steps": "Slice beets and mix with lemon."},
            {"title": "Beet Juice", "steps": "Blend beetroot with water."}
        ]
    },
    "bell pepper": {
        "nutrition": {"Vitamin C": "High", "Calories": "20 kcal"},
        "recipes": [
            {"title": "Pepper Stir Fry", "steps": "Stir-fry sliced peppers."},
            {"title": "Pepper Sandwich", "steps": "Add sliced peppers to sandwich."}
        ]
    },
    "berries": {
        "nutrition": {"Antioxidants": "High", "Calories": "57 kcal"},
        "recipes": [
            {"title": "Berry Bowl", "steps": "Mix berries with yogurt."},
            {"title": "Berry Shake", "steps": "Blend berries with milk."}
        ]
    },
    "blackberry": {
        "nutrition": {"Fiber": "5 g", "Vitamin C": "35% DV"},
        "recipes": [
            {"title": "Blackberry Mix", "steps": "Add to yogurt and mix."},
            {"title": "Blackberry Syrup", "steps": "Boil berries with sugar."}
        ]
    },
    "brocolli": {
        "nutrition": {"Vitamin K": "High", "Calories": "34 kcal"},
        "recipes": [
            {"title": "Boiled Broccoli", "steps": "Boil florets for 5 minutes."},
            {"title": "Broccoli Salad", "steps": "Mix boiled broccoli with mayo."}
        ]
    },
    "cabbage": {
        "nutrition": {"Calories": "25 kcal", "Fiber": "2.5 g"},
        "recipes": [
            {"title": "Cabbage Stir Fry", "steps": "Stir-fry shredded cabbage."},
            {"title": "Raw Cabbage Salad", "steps": "Mix cabbage with lemon and salt."}
        ]
    },
    "cantaloupe": {
        "nutrition": {"Vitamin A": "68% DV", "Water": "90%"},
        "recipes": [
            {"title": "Melon Bowl", "steps": "Serve diced melon."},
            {"title": "Melon Juice", "steps": "Blend melon with water."}
        ]
    },
    "capsicum": {
        "nutrition": {"Vitamin C": "High", "Calories": "20 kcal"},
        "recipes": [
            {"title": "Capsicum Fry", "steps": "Saute sliced capsicum."},
            {"title": "Capsicum Topping", "steps": "Use as pizza topping."}
        ]
    },
    "carrot": {
        "nutrition": {"Vitamin A": "334% DV", "Calories": "41 kcal"},
        "recipes": [
            {"title": "Carrot Salad", "steps": "Grate carrot and mix with lemon."},
            {"title": "Carrot Stir Fry", "steps": "Cook sliced carrots in oil."}
        ]
    },
    "cauliflower": {
        "nutrition": {"Calories": "25 kcal", "Vitamin C": "48% DV"},
        "recipes": [
            {"title": "Cauliflower Fry", "steps": "Fry florets with spices."},
            {"title": "Cauliflower Soup", "steps": "Blend boiled cauliflower."}
        ]
    },
    "cherry": {
        "nutrition": {"Calories": "50 kcal", "Vitamin C": "12% DV"},
        "recipes": [
            {"title": "Cherry Mix", "steps": "Add cherries to salad."},
            {"title": "Cherry Juice", "steps": "Blend cherries with sugar."}
        ]
    },
    "chilli pepper": {
        "nutrition": {"Vitamin C": "High", "Heat": "High"},
        "recipes": [
            {"title": "Chili Pickle", "steps": "Mix chilies with lemon & salt."},
            {"title": "Spicy Tadka", "steps": "Fry chilies in oil for aroma."}
        ]
    },
    "corn": {
        "nutrition": {"Calories": "86 kcal", "Carbs": "19 g"},
        "recipes": [
            {"title": "Boiled Corn", "steps": "Boil corn with salt."},
            {"title": "Masala Corn", "steps": "Mix boiled corn with butter & masala."}
        ]
    },
    "courgette": {
        "nutrition": {"Calories": "17 kcal", "Water": "94%"},
        "recipes": [
            {"title": "Zucchini Fry", "steps": "Fry sliced courgette."},
            {"title": "Zucchini Salad", "steps": "Mix raw slices with lemon."}
        ]
    },
    "cucumber": {
        "nutrition": {"Water": "96%", "Calories": "15 kcal"},
        "recipes": [
            {"title": "Cucumber Salad", "steps": "Mix cucumber & salt."},
            {"title": "Cucumber Raita", "steps": "Mix grated cucumber with curd."}
        ]
    },
    "dates": {
        "nutrition": {"Calories": "277 kcal", "Fiber": "7 g"},
        "recipes": [
            {"title": "Date Shake", "steps": "Blend dates with milk."},
            {"title": "Stuffed Dates", "steps": "Fill dates with nuts."}
        ]
    },
    "dragon fruit": {
        "nutrition": {"Vitamin C": "3% DV", "Calories": "60 kcal"},
        "recipes": [
            {"title": "Dragon Fruit Bowl", "steps": "Serve cubed fruit."},
            {"title": "DF Smoothie", "steps": "Blend dragon fruit with yogurt."}
        ]
    },
    "eggplant": {
        "nutrition": {"Calories": "25 kcal", "Fiber": "3 g"},
        "recipes": [
            {"title": "Eggplant Fry", "steps": "Fry eggplant slices."},
            {"title": "Mashed Eggplant", "steps": "Roast and mash eggplant."}
        ]
    },
    "fig": {
        "nutrition": {"Calories": "74 kcal", "Fiber": "2.9 g"},
        "recipes": [
            {"title": "Fig Salad", "steps": "Mix figs with nuts."},
            {"title": "Fig Shake", "steps": "Blend figs with milk."}
        ]
    },
    "garlic": {
        "nutrition": {"Allicin": "High", "Calories": "149 kcal"},
        "recipes": [
            {"title": "Garlic Tadka", "steps": "Fry chopped garlic in oil."},
            {"title": "Garlic Spread", "steps": "Mix garlic with butter."}
        ]
    },
    "ginger": {
        "nutrition": {"Anti-inflammatory": "High"},
        "recipes": [
            {"title": "Ginger Tea", "steps": "Boil ginger in water."},
            {"title": "Ginger Stir Fry", "steps": "Add grated ginger to veggies."}
        ]
    },
    "grapes": {
        "nutrition": {"Calories": "69 kcal", "Carbs": "18 g"},
        "recipes": [
            {"title": "Grape Bowl", "steps": "Serve grapes directly."},
            {"title": "Grape Juice", "steps": "Blend grapes and strain."}
        ]
    },
    "jalepeno": {
        "nutrition": {"Heat": "Medium", "Vitamin C": "High"},
        "recipes": [
            {"title": "Jalapeño Topping", "steps": "Add slices to pizza."},
            {"title": "Stuffed Jalapeño", "steps": "Fill with cheese and bake."}
        ]
    },
    "kiwi": {
        "nutrition": {"Vitamin C": "92% DV", "Calories": "61 kcal"},
        "recipes": [
            {"title": "Kiwi Bowl", "steps": "Serve sliced kiwi."},
            {"title": "Kiwi Smoothie", "steps": "Blend kiwi with yogurt."}
        ]
    },
    "lemon": {
        "nutrition": {"Vitamin C": "50% DV", "Calories": "29 kcal"},
        "recipes": [
            {"title": "Lemon Water", "steps": "Mix lemon in warm water."},
            {"title": "Lemon Dressing", "steps": "Mix lemon with olive oil."}
        ]
    },
    "lettuce": {
        "nutrition": {"Calories": "15 kcal", "Vitamin K": "36% DV"},
        "recipes": [
            {"title": "Fresh Salad", "steps": "Serve lettuce with dressing."},
            {"title": "Lettuce Roll", "steps": "Use as wrap for veggies."}
        ]
    },
    "lychee": {
        "nutrition": {"Vitamin C": "119% DV", "Calories": "66 kcal"},
        "recipes": [
            {"title": "Lychee Juice", "steps": "Blend lychee with water."},
            {"title": "Lychee Bowl", "steps": "Serve peeled lychee."}
        ]
    },
    "mango": {
        "nutrition": {"Calories": "60 kcal", "Vitamin C": "44% DV"},
        "recipes": [
            {"title": "Mango Lassi", "steps": "Blend mango with curd."},
            {"title": "Mango Salsa", "steps": "Mix mango with onion & lime."}
        ]
    },
    "nectarine": {
        "nutrition": {"Calories": "44 kcal", "Vitamin C": "9% DV"},
        "recipes": [
            {"title": "Nectarine Slices", "steps": "Serve as is."},
            {"title": "Quick Nectarine Salad", "steps": "Mix slices with honey."}
        ]
    },
    "olive": {
        "nutrition": {"Healthy Fats": "Good", "Calories": "115 kcal"},
        "recipes": [
            {"title": "Olive Snack", "steps": "Serve olives directly."},
            {"title": "Olive Spread", "steps": "Blend olives with oil."}
        ]
    },
    "onion": {
        "nutrition": {"Antioxidants": "High", "Calories": "40 kcal"},
        "recipes": [
            {"title": "Onion Fry", "steps": "Fry sliced onions."},
            {"title": "Onion Salad", "steps": "Mix sliced onion with lemon."}
        ]
    },
    "orange": {
        "nutrition": {"Vitamin C": "88% DV", "Calories": "47 kcal"},
        "recipes": [
            {"title": "Orange Juice", "steps": "Squeeze fresh oranges."},
            {"title": "Orange Bowl", "steps": "Serve peeled orange slices."}
        ]
    },
    "paprika": {
        "nutrition": {"Vitamin A": "High", "Calories": "282 kcal"},
        "recipes": [
            {"title": "Paprika Rub", "steps": "Mix paprika with salt."},
            {"title": "Paprika Oil", "steps": "Heat oil with paprika."}
        ]
    },
    "passion": {
        "nutrition": {"Vitamin C": "30% DV", "Fiber": "10 g"},
        "recipes": [
            {"title": "Passion Juice", "steps": "Blend with water and strain."},
            {"title": "Passion Topping", "steps": "Add pulp to yogurt."}
        ]
    },
    "pawpaw": {
        "nutrition": {"Vitamin A": "22% DV", "Calories": "43 kcal"},
        "recipes": [
            {"title": "Pawpaw Bowl", "steps": "Serve sliced pawpaw."},
            {"title": "Pawpaw Shake", "steps": "Blend with milk."}
        ]
    },
    "peach": {
        "nutrition": {"Calories": "39 kcal", "Vitamin C": "6% DV"},
        "recipes": [
            {"title": "Peach Slices", "steps": "Serve fresh slices."},
            {"title": "Peach Smoothie", "steps": "Blend peach with yogurt."}
        ]
    },
    "pear": {
        "nutrition": {"Calories": "57 kcal", "Fiber": "3.1 g"},
        "recipes": [
            {"title": "Pear Salad", "steps": "Mix pear chunks with nuts."},
            {"title": "Pear Shake", "steps": "Blend pear with milk."}
        ]
    },
    "peas": {
        "nutrition": {"Protein": "5 g", "Calories": "81 kcal"},
        "recipes": [
            {"title": "Peas Stir Fry", "steps": "Cook peas with spices."},
            {"title": "Peas Mix", "steps": "Add boiled peas to salads."}
        ]
    },
    "pepino": {
        "nutrition": {"Water": "92%", "Calories": "22 kcal"},
        "recipes": [
            {"title": "Pepino Slices", "steps": "Serve fresh slices."},
            {"title": "Pepino Salad", "steps": "Mix slices with lemon."}
        ]
    },
    "pineapple": {
        "nutrition": {"Calories": "50 kcal", "Vitamin C": "47% DV"},
        "recipes": [
            {"title": "Pineapple Juice", "steps": "Blend pineapple pieces."},
            {"title": "Pineapple Chunks", "steps": "Serve diced pineapple."}
        ]
    },
    "plum": {
        "nutrition": {"Calories": "46 kcal", "Vitamin C": "10% DV"},
        "recipes": [
            {"title": "Plum Mix", "steps": "Add sliced plums to salad."},
            {"title": "Plum Shake", "steps": "Blend plums with yogurt."}
        ]
    },
    "pomegranate": {
        "nutrition": {"Antioxidants": "High", "Calories": "83 kcal"},
        "recipes": [
            {"title": "Pom Bowl", "steps": "Serve seeds directly."},
            {"title": "Pom Juice", "steps": "Crush seeds and strain."}
        ]
    },
    "potato": {
        "nutrition": {"Calories": "77 kcal", "Carbs": "17 g"},
        "recipes": [
            {"title": "Boiled Potato", "steps": "Boil potatoes with salt."},
            {"title": "Potato Fry", "steps": "Shallow fry sliced potatoes."}
        ]
    },
    "pumpkin": {
        "nutrition": {"Vitamin A": "170% DV", "Calories": "26 kcal"},
        "recipes": [
            {"title": "Pumpkin Stir Fry", "steps": "Cook pumpkin cubes."},
            {"title": "Pumpkin Mash", "steps": "Boil and mash pumpkin."}
        ]
    },
    "raddish": {
        "nutrition": {"Vitamin C": "17% DV", "Calories": "16 kcal"},
        "recipes": [
            {"title": "Radish Salad", "steps": "Grate radish and serve."},
            {"title": "Radish Fry", "steps": "Saute sliced radish."}
        ]
    },
    "soy beans": {
        "nutrition": {"Protein": "36 g", "Fiber": "9 g"},
        "recipes": [
            {"title": "Boiled Soybeans", "steps": "Boil until soft."},
            {"title": "Soybean Salad", "steps": "Mix boiled soybeans with veggies."}
        ]
    },
    "spinach": {
        "nutrition": {"Iron": "15% DV", "Vitamin A": "186% DV"},
        "recipes": [
            {"title": "Spinach Stir Fry", "steps": "Cook spinach with garlic."},
            {"title": "Spinach Soup", "steps": "Blend boiled spinach."}
        ]
    },
    "strawberry": {
        "nutrition": {"Vitamin C": "97% DV", "Calories": "32 kcal"},
        "recipes": [
            {"title": "Strawberry Shake", "steps": "Blend strawberries with milk."},
            {"title": "Strawberry Bowl", "steps": "Serve sliced strawberries."}
        ]
    },
    "sugar apple": {
        "nutrition": {"Calories": "94 kcal", "Fiber": "4.4 g"},
        "recipes": [
            {"title": "Sugar Apple Pulp", "steps": "Scoop and eat."},
            {"title": "Sugar Apple Shake", "steps": "Blend pulp with milk."}
        ]
    },
    "sweetcorn": {
        "nutrition": {"Carbs": "19 g", "Calories": "86 kcal"},
        "recipes": [
            {"title": "Sweet Corn Salad", "steps": "Mix boiled corn with veggies."},
            {"title": "Corn Cup", "steps": "Add butter and salt to boiled corn."}
        ]
    },
    "sweetpotato": {
        "nutrition": {"Vitamin A": "284% DV", "Calories": "86 kcal"},
        "recipes": [
            {"title": "Boiled Sweet Potato", "steps": "Boil and peel."},
            {"title": "Sweet Potato Fry", "steps": "Fry thin slices."}
        ]
    },
    "tangarine": {
        "nutrition": {"Vitamin C": "44% DV", "Calories": "53 kcal"},
        "recipes": [
            {"title": "Tangerine Bowl", "steps": "Serve peeled slices."},
            {"title": "Tangerine Juice", "steps": "Blend tangerine with water."}
        ]
    },
    "tomato": {
        "nutrition": {"Vitamin C": "21% DV", "Calories": "18 kcal"},
        "recipes": [
            {"title": "Tomato Salad", "steps": "Slice tomatoes and serve."},
            {"title": "Tomato Juice", "steps": "Blend tomatoes and strain."}
        ]
    },
    "turnip": {
        "nutrition": {"Calories": "28 kcal", "Vitamin C": "21% DV"},
        "recipes": [
            {"title": "Turnip Fry", "steps": "Stir-fry sliced turnip."},
            {"title": "Turnip Salad", "steps": "Mix grated turnip with lemon."}
        ]
    },
    "watermelon": {
        "nutrition": {"Water": "92%", "Calories": "30 kcal"},
        "recipes": [
            {"title": "Watermelon Juice", "steps": "Blend watermelon chunks."},
            {"title": "Watermelon Cubes", "steps": "Serve chilled cubes."}
        ]
    }
}
DEFAULT_INFO = {
    "nutrition": {"Info": "No data available"},
    "recipes": [{"title": "No recipe", "steps": "No instructions available"}]
}

# ============================================================
#              IMAGE PREPROCESSING
# ============================================================
def preprocess_image_bytes(img_bytes, auto_enhance=True):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    if auto_enhance:
        try:
            img = ImageOps.autocontrast(img)
        except:
            pass

    img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = tf.keras.utils.img_to_array(img_resized)
    arr = np.expand_dims(arr, 0)

    return arr, img_resized


# ============================================================
#        IMAGE QUALITY + SPOILAGE DETECTION (NEW FEATURE)
# ============================================================
def analyze_quality_and_spoilage(pil_image):
    np_img = np.array(pil_image)

    # ---- SHARPNESS ----
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    sharpness_raw = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_norm = min(100.0, (sharpness_raw / 300.0) * 100.0)

    # ---- BRIGHTNESS ----
    brightness_norm = (gray.mean() / 255.0) * 100.0

    # ---- EDGE DENSITY ----
    edges = cv2.Canny(gray, 100, 200)
    edge_density = (np.count_nonzero(edges) / edges.size) * 100.0

    # ---- QUALITY SCORE ----
    brightness_penalty = abs(brightness_norm - 55)
    quality = (
        0.6 * sharpness_norm +
        0.2 * (100 - brightness_penalty) +
        0.2 * max(0.0, 100 - abs(edge_density - 40))
    )
    quality = max(0.0, min(100.0, quality))

    # ---- SPOILAGE ----
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    dark_ratio = (v < 60).mean()
    bruise_ratio = ((v < 110) & (s < 50)).mean()
    spoilage_score = float((dark_ratio * 0.6 + bruise_ratio * 0.4) * 100.0)

    if spoilage_score < 15:
        status = "Fresh"
        note = "Looks fresh with no major discolored areas."
    elif spoilage_score < 35:
        status = "Slightly Aged"
        note = "Some dark or dull patches detected."
    else:
        status = "Possibly Spoiled"
        note = "Many dark/saturated patches — may be spoiled."

    return (
        {
            "overall": round(quality, 1),
            "sharpness": round(sharpness_norm, 1),
            "brightness": round(brightness_norm, 1),
            "edge_density": round(edge_density, 1),
        },
        {
            "status": status,
            "score": round(spoilage_score, 1),
            "note": note
        }
    )


# ============================================================
#               NUTRITION LOOKUP
# ============================================================
def lookup_nutrition_recipes(label):
    return NUTRITION_RECIPES.get(label.lower(), DEFAULT_INFO)


# ============================================================
#                   ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    uploaded_files = request.files.getlist("images")
    if not uploaded_files:
        return "No images uploaded", 400

    results = []

    for file in uploaded_files:
        try:
            img_bytes = file.read()

            # Original image
            pil_original = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Preprocessed image
            arr, pil_processed = preprocess_image_bytes(img_bytes)

            # Edge detection
            cv_img = cv2.cvtColor(np.array(pil_original), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(cv_img, 100, 200)
            pil_edges = Image.fromarray(edges)

            # Convert PIL to base64
            def to_b64(img):
                buff = io.BytesIO()
                img.save(buff, format="JPEG")
                return base64.b64encode(buff.getvalue()).decode("utf-8")

            original_b64 = to_b64(pil_original)
            processed_b64 = to_b64(pil_processed)
            edge_b64 = to_b64(pil_edges)

            # Predict
            preds = model.predict(arr)
            probs = tf.nn.softmax(preds[0]).numpy()
            top3_idx = probs.argsort()[-3:][::-1]
            top3 = [(data_cat[i], round(float(probs[i]*100), 2)) for i in top3_idx]

            main_label = top3[0][0]
            confidence = top3[0][1]

            # Nutrition & recipes
            info = lookup_nutrition_recipes(main_label)

            # NEW FEATURE: Image quality + spoilage
            quality_info, spoilage_info = analyze_quality_and_spoilage(pil_original)

            # Final result object
            results.append({
                "prediction": main_label,
                "confidence": confidence,
                "top3": top3,
                "image_b64": base64.b64encode(img_bytes).decode("utf-8"),

                "original_b64": original_b64,
                "processed_b64": processed_b64,
                "edge_b64": edge_b64,

                "nutrition": info["nutrition"],
                "recipes": info["recipes"],

                "quality": quality_info,
                "spoilage": spoilage_info
            })

        except Exception as e:
            results.append({"error": str(e)})

    app.latest_results = results
    return render_template("result.html", results=results)


@app.route("/download_csv")
def download_csv():
    results = getattr(app, "latest_results", [])
    if not results:
        return "No results", 404

    def generate():
        yield "prediction,confidence,top3\n"
        for r in results:
            t = ";".join([f"{a}:{b}" for a,b in r["top3"]])
            yield f"{r['prediction']},{r['confidence']},{t}\n"

    return Response(
        generate(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=results.csv"}
    )


# ============================================================
#                    RUN APP
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
