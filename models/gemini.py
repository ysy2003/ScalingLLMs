from google import genai
from google.genai.types import HttpOptions, Part
import dotenv
dotenv.load_dotenv()
PROJECT_ID = "scallingllms-project-web2ui"
client = genai.Client(project=PROJECT_ID,
    http_options=HttpOptions(api_version="v1"))
# example of using the gemini model to generate content from an image
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        "What is shown in this image?",
        Part.from_uri(
            file_uri="gs://scallingllms-design2code-dataset/dataset/test.png",
            mime_type="image/jpeg",
        ),
    ],
)
print(response.text)