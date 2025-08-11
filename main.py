import base64
import io
import logging
import os
from textwrap import dedent
import requests
from crewai import Agent, Crew, Task, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool
from flask import Flask, render_template, request, send_file
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Set up Google Gemini API
google_api_key = os.environ['GOOGLE_API_KEY']
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=google_api_key,
)

serper_api_key = os.environ['SERPER_API_KEY']
search_tool = SerperDevTool()

# Set up OpenAI API for DALL-E
openai_api_key = os.environ['OPENAI_API_KEY']

web_search_tool = WebsiteSearchTool(config=dict(
    llm=dict(
        provider="google",
        config=dict(model="gemini-1.5-flash", ),
    ),
    embedder=dict(
        provider="google",
        config=dict(
            model="models/embedding-001",
            task_type="retrieval_document",
        ),
    ),
))

# Define agents
writer_agent = Agent(name="Lekhak",
                     role="Writer",
                     goal=dedent("""
        Write high-quality content based on user prompts also the recent events or trending activities must be taken into account while generating any type of content. Make sure the writing is good and follows what people ask for. Also, include information about what's happening now or what people are talking about.
        """),
                     verbose=True,
                     memory=True,
                     backstory=dedent("""
        You are an experienced writer capable of creating various types of content including Blog posts, Articles, Newsletters, Youtube Scripts, Facebook Scripts, Instagram Script and Stories. 
        """),
                     tools=[search_tool, web_search_tool],
                     llm=gemini_llm,
                     allow_delegation=True)


def generate_image(prompt):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {google_api_key}"
        }

        payload = {
            "model": gemini_llm,
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024"
        }

        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers=headers,
            json=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        image_url = response.json()['data'][0]['url']
        image_response = requests.get(image_url)
        image_response.raise_for_status()

        return base64.b64encode(image_response.content).decode('utf-8')
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error generating image: {str(e)}")
        if response.status_code != 200:
            app.logger.error(f"API Response: {response.text}")
        return None


image_generation_tool = Tool(
    name="Image Generation",
    func=generate_image,
    description=dedent(
        """Generates images based on text prompts using gemini model."""),
    model=gemini_llm,
)

image_creator_agent = Agent(
    role='Image Creator',
    goal=dedent("""Create images based on user prompts"""),
    backstory=dedent(
        """You are a skilled artist capable of generating images from textual descriptions."""
    ),
    tools=[image_generation_tool],
    verbose=True,
    llm=gemini_llm  # Use Gemini for the image creator agent as well
)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image = None
    error = None
    if request.method == 'POST':
        prompt = request.form['prompt']
        content_type = request.form['content_type']

        app.logger.debug(f"Received request - Prompt: {prompt}")

        if content_type == 'image':
            try:
                app.logger.debug("Attempting to generate image...")
                image_data = generate_image(prompt)
                if image_data:
                    image = image_data
                    app.logger.debug("Image generated successfully")
                else:
                    error = "Failed to generate image"
                    app.logger.error("Image generation returned None")
            except Exception as e:
                error = f"Error generating image: {str(e)}"
                app.logger.error(f"Image generation error: {str(e)}",
                                 exc_info=True)

        if content_type == 'text':
            try:
                task = Task(
                    description=
                    f"Create {content_type} content based on the following prompt: {prompt}",
                    tools=[search_tool, web_search_tool],
                    agent=writer_agent,
                    expected_output=dedent("""
                        The final output should have creative, engaging content which includes latest news, events, trending activities and the required content based on the given prompt.
                        """),
                )
                crew = Crew(
                    agents=[writer_agent],
                    tasks=[task],
                    process=Process.sequential,
                )
                result = crew.kickoff()
                app.logger.debug(f"Generated text content: {result}")
            except Exception as e:
                error = f"Error generating text: {str(e)}"
                app.logger.error(f"Text generation error: {str(e)}",
                                 exc_info=True)

        elif content_type == 'image':
            try:
                image_data = generate_image(prompt)
                if image_data:
                    image = image_data
                    app.logger.debug("Image generated successfully")
                else:
                    error = "Failed to generate image"
                    app.logger.error("Image generation failed")
            except Exception as e:
                error = f"Error generating image: {str(e)}"
                app.logger.error(f"Image generation error: {str(e)}",
                                 exc_info=True)

    app.logger.debug(
        f"Rendering template - Image: {'Present' if image else 'None'}, Error: {error}"
    )
    app.logger.debug(f"API Key present: {'Yes' if openai_api_key else 'No'}")
    return render_template('index.html',
                           result=result,
                           image=image,
                           error=error)


@app.route('/download-image')
def download_image():
    image_data = base64.b64decode(request.args.get('image'))
    return send_file(io.BytesIO(image_data),
                     mimetype='image/png',
                     as_attachment=True,
                     download_name='generated_image.png')


if __name__ == '__main__':
    app.run(debug=True)
