# Importing necessary libraries
import streamlit as st
import PyPDF2
import io
import openai
import docx2txt
import pyperclip
import os  # Added the OS module to work with file directories

# Setting up OpenAI API key
openai_api_key = st.secrets['OPENAI_API_KEY'] # This takes the API key from the secrets.toml file
# openai.api_key = st.sidebar.text_input('OpenAI API Key', type='password') # Prompt to user for API key

# Defining a function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text

# Function to list PDF files in a directory
def list_pdf_files(directory):
    pdf_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory, filename))
    return pdf_files

# Updating function to generate questions from text using OpenAI's updated API
def get_questions_from_gpt(text):
    prompt = text[:4096]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5, 
        max_tokens=30
    )
    return response['choices'][0]['message']['content'].strip()

# Updating function to generate answers to a question using OpenAI's updated API
def get_answers_from_gpt(text, question):
    prompt = text[:4096] + "\nQuestion: " + question + "\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6, 
        max_tokens=2000
    )
    return response['choices'][0]['message']['content'].strip()

# Defining the main function of the Streamlit app
def main():
    # enter the paths where .pdf files are placed like (pdf_files/)
    st.title("🤷‍♂️ Ask Questions From Documents") 
    
    # Get the folder containing PDF files using folder input
    pdf_folder = st.text_input("Enter the folder path containing PDF files:", placeholder='Enter pdf_files/ as a sample')
    
    if pdf_folder and os.path.isdir(pdf_folder):
        pdf_files = list_pdf_files(pdf_folder)
        
        if not pdf_files:
            st.warning("No PDF files found in the specified folder.")
        else:
            st.info(f"Number of PDF files found: {len(pdf_files)}")
            
            # Select PDF file
            selected_pdf = st.selectbox("Select a PDF file", pdf_files)
            st.info(f"Selected PDF: {selected_pdf}")
            
            # Extract text from the selected PDF
            text = extract_text_from_pdf(selected_pdf)
            
            # It's a better idea to split the text into smaller chunks if needed
            # For example, split the text into paragraphs or sentences
            st.write("<span style='color:brown; font-size:20px'><b>The following questions are extracted from file automatically:</b></span>", unsafe_allow_html=True) 
            
            # Generating multiple questions from the extracted text using GPT-4
            num_questions = 5  # Set the number of questions you want to generate
            questions = []
            for _ in range(num_questions):
                question = get_questions_from_gpt(text)
                questions.append(question)

            # Display the generated questions
            for i, question in enumerate(questions):
                st.write(f"Question {i+1}: {question}")
            
            # Copy the generated questions to the clipboard
            #for question in questions:
            #    pyperclip.copy(question)
            #    st.success("Questions copied to clipboard!")    
            
            # Generating single question from the extracted text using GPT-4
            # question = get_questions_from_gpt(text)
            # st.write("Question: " + question)
            
            # Creating a text input for the user to ask a question
            st.write("<span style='color:green; font-size:20px'><b>If you have more questions</b></span>", unsafe_allow_html=True)
            user_question = st.text_input("Ask a question about the document")
            
            if user_question:
                # Generating an answer to the user's question using GPT-4
                answer = get_answers_from_gpt(text, user_question)
                st.write("Answer: " + answer)
                if st.button("Copy Answer Text"):
                    pyperclip.copy(answer)
                    st.success("Answer text copied to clipboard!")

# Running the main function if the script is being run directly
if __name__ == '__main__':
    main()

# Side bar content
st.sidebar.markdown("---")
st.sidebar.subheader("Credit to : Dr. Ammar Tufail\nCEO of [Codanics.com](https://codanics.com)")
# add a youtube video
st.sidebar.video("https://youtu.be/omk5b1m2h38")
st.sidebar.markdown("---")
# add social contact info
st.sidebar.markdown("Created & Designed by : [Shahid Umar](mailto:shahidcontacts@gmail.com)")
st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Profile-red?style=for-the-badge&logo=github)](https://github.com/Shahid-Umar)&emsp;")
st.sidebar.markdown("[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/shahidumar80)&emsp;")
st.sidebar.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-green?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/shahidumar/)&emsp;")
st.sidebar.markdown("[![Facebook](https://img.shields.io/badge/Facebook-Profile-pink?style=for-the-badge&logo=facebook)](https://www.facebook.com/shahidumar80)&emsp;")
st.sidebar.markdown("[![Twitter](https://img.shields.io/badge/Twitter-Profile-brown?style=for-the-badge&logo=twitter)](https://twitter.com/shahidumar80/)")
