from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import pandas as pd
import torch
import torch.nn.functional as F
from open_clip import create_model_and_transforms, get_tokenizer

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and preprocess function
model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
model.eval()

# Load tokenizer
tokenizer = get_tokenizer('ViT-B-32')

# Load embeddings
df = pd.read_pickle('image_embeddings.pickle')
embeddings = torch.stack([torch.tensor(emb) for emb in df['embedding'].values])

# Serve images from the `coco_images_resized` folder
@app.route('/coco_images_resized/<path:filename>')
def serve_image(filename):
    return send_from_directory('coco_images_resized', filename)

# Helper function to find top K similar images
def find_top_k_similar(query_embedding, embeddings, df, k=5):
    similarities = F.cosine_similarity(query_embedding, embeddings)
    top_k_indices = similarities.topk(k).indices
    results = [
        {
            "file_name": f"coco_images_resized/{df.iloc[int(i)]['file_name']}",  # Update path here
            "similarity": similarities[i].item(),
        }
        for i in top_k_indices
    ]
    return results

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle queries
        query_type = request.form.get("query_type")
        hybrid_weight = float(request.form.get("hybrid_weight", 0.5))

        query_embedding = None
        if query_type == "text":
            text_query = request.form.get("text_query", "")
            tokenized_text = tokenizer([text_query])  # Correct tokenizer usage
            query_embedding = F.normalize(model.encode_text(tokenized_text))
        elif query_type == "image":
            if "image_query" in request.files:
                image_file = request.files["image_query"]
                image = preprocess(Image.open(image_file)).unsqueeze(0)
                query_embedding = F.normalize(model.encode_image(image))
        elif query_type == "hybrid":
            text_query = request.form.get("text_query", "")
            tokenized_text = tokenizer([text_query])
            text_embedding = F.normalize(model.encode_text(tokenized_text))

            if "image_query" in request.files:
                image_file = request.files["image_query"]
                image = preprocess(Image.open(image_file)).unsqueeze(0)
                image_embedding = F.normalize(model.encode_image(image))

                query_embedding = F.normalize(hybrid_weight * text_embedding + (1 - hybrid_weight) * image_embedding)

        # Find top 5 similar
        results = find_top_k_similar(query_embedding, embeddings, df, k=5)

        return render_template("index.html", results=results)

    return render_template("index.html", results=None)

if __name__ == "__main__":
    app.run(debug=True, port=3000)
