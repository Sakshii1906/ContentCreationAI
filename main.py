from flask import Flask, render_template, request, send_file
import google.generativeai as genai
from stability_sdk import client as stability_client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import base64
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configure Stability AI
stability_api = stability_client.StabilityInference(
    key=os.getenv("STABILITY_API_KEY"),
    verbose=True,
)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_base64 = None
    error = None

    if request.method == "POST":
        prompt = request.form.get("prompt")
        content_type = request.form.get("content_type")

        try:
            if content_type == "text":
                # Text generation with Gemini
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                result = response.text

            elif content_type == "image":
                # Image generation with Stability AI
                answers = stability_api.generate(
                    prompt=prompt,
                    width=512,
                    height=512,
                    steps=30,
                    cfg_scale=8.0,
                )

                for resp in answers:
                    for artifact in resp.artifacts:
                        if artifact.type == generation.ARTIFACT_IMAGE:
                            image_base64 = base64.b64encode(artifact.binary).decode("utf-8")
                            with open("generated_image.png", "wb") as f:
                                f.write(artifact.binary)

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        result=result,
        image=image_base64,
        error=error
    )

@app.route("/download_image")
def download_image():
    try:
        return send_file("generated_image.png", as_attachment=True)
    except FileNotFoundError:
        return "No image to download", 404

if __name__ == "__main__":
    app.run(debug=True)
