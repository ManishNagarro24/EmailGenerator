import os
import openai
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.mapreduce import MapReduceChain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

st.set_page_config(page_title="Nagarro", page_icon="img/Nagarro_logo.png",)

st.markdown('''<style>.css-1egvi7u {margin-top: -4rem;}</style>''',
    unsafe_allow_html=True)

st.markdown('''<style>.css-znku1x a {color: #9d03fc;}</style>''',
    unsafe_allow_html=True)  # darkmode
st.markdown('''<style>.css-znku1x a {color: #9d03fc;}</style>''',
    unsafe_allow_html=True)  # lightmode

st.markdown('''<style>.css-qrbaxs {min-height: 0.0rem;}</style>''',
    unsafe_allow_html=True)

st.markdown('''<style>.stSpinner > div > div {border-top-color: #9d03fc;}</style>''',
    unsafe_allow_html=True)

st.markdown('''<style>.css-15tx938{min-height: 0.0rem;}</style>''',
    unsafe_allow_html=True)

hide_decoration_bar_style = '''<style>header {visibility: hidden;}</style>'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

hide_streamlit_footer = """<style>#MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_footer, unsafe_allow_html=True)

# Connect to OpenAI GPT-3

openai.api_type = "azure"
openai.api_base = "https://emailgeneratordemo.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "4461d4ebc79a45bca18557145962a4f3"
print('Till Here')
def gen_mail_contents(email_contents):
    
    for topic in range(len(email_contents)):
        input_text = email_contents[topic]

        openaiq = OpenAI(temperature=.7,openai_api_key="4461d4ebc79a45bca18557145962a4f3",deployment_id="EmailGeneratorDemo02")
    
        prompt = """Write an email content based on the following information provided

Context: This email is regarding the 

Question: Which libraries and model providers offer LLMs?

Answer: """
        print(openaiq(prompt))                                            
        map_prompt = """Below is a section of a website about {prospect}
       Write a concise summary about {prospect}. If the information is not about {prospect}, exclude it from your summary.{text}
% CONCISE SUMMARY:"""
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "prospect"])

        combine_prompt = """
Your goal is to write a personalized outbound email for Campaign {Reason} by {company} to {prospect}.

A good email is personalized and combines information about the two companies on how they can help each other.
Be sure to use value selling: A sales methodology that focuses on how your product or service will provide value to the customer instead of focusing on price or solution.

% INFORMATION ABOUT {company}:
{company_information}

% INFORMATION ABOUT {prospect}:
{text}

% INCLUDE THE FOLLOWING PIECES IN YOUR RESPONSE:
- Start the email with the sentence: "We love that {prospect} helps teams..." then insert what they help teams do.
- The sentence: "We can help you do XYZ by ABC" Replace XYZ with what {prospect} does and ABC with what {company} does 
- A 1-2 sentence description about {company}, be brief
- End your email with a call-to-action such as asking them to set up time to talk more

% YOUR RESPONSE:
"""
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["Reason", "company", "prospect", \
                                                                         "text", "company_information"])

        chain = load_summarize_chain(openaiq,
                             chain_type="map_reduce",
                             map_prompt=map_prompt_template,
                             combine_prompt=combine_prompt_template,
                             verbose=True
                            )
        text_splitter = CharacterTextSplitter()
        with open("./BlackBaud.txt") as f:
            state_of_the_union = f.read()
            texts = text_splitter.split_text(state_of_the_union)
       
        docs = [Document(page_content=t) for t in texts[:3]]

        output = chain({ "input_documents": docs,
                "company": "Nagarro", \
                "company_information" : "Nagarro is a leading global software engineering and technology consulting company that specializes in delivering innovative solutions and services to businesses across various industries. With a strong focus on digital transformation, Nagarro helps organizations harness the power of technology to drive growth, enhance customer experiences, and optimize operational efficiency. Their team of highly skilled professionals brings together expertise in areas such as software development, cloud computing, artificial intelligence, data analytics, and agile methodologies. Nagarro's client-centric approach ensures that they understand the unique challenges and requirements of each business, enabling them to deliver tailor-made solutions that meet their specific needs. Committed to quality and innovation, Nagarro constantly pushes boundaries to deliver cutting-edge solutions that enable businesses to stay ahead in the rapidly evolving digital landscape. Collaboration and teamwork are at the core of Nagarro's culture, fostering an environment where employees are encouraged to think creatively, share ideas, and work together to deliver exceptional results. With a global presence and extensive industry experience, Nagarro serves clients from around the world, ranging from startups to multinational corporations, helping them achieve their digital goals and drive success.",
                "Reason" : "Uneducated India", \
                "prospect" : "Blackbaud"
               })
        langout=output['output_text']



        rephrased_content = openai.Completion.create(
            deployment_id="EmailGeneratorDemo02",
            prompt=f"Rewrite the text to be elaborate and polite.\nAbbreviations need to be replaced.\nText: {input_text}\nRewritten text:",
            
            temperature=0.8,
            max_tokens=len(input_text)*3,
            top_p=0.8,
            best_of=2,
            frequency_penalty=0.0,
            presence_penalty=0.0)

        
        email_contents[topic] = rephrased_content.get("choices")[0]['text']
    return email_contents


def gen_mail_format(sender, recipient, style, email_contents):
    
    email_contents = gen_mail_contents(email_contents)
    

    contents_str, contents_length = "", 0
    for topic in range(len(email_contents)):  
        contents_str = contents_str + f"\nContent{topic+1}: " + email_contents[topic]
        contents_length += len(email_contents[topic]) 

    email_final_text = openai.Completion.create(
        deployment_id="EmailGeneratorDemo02",
       
        prompt=f"""Write a email body for {style} and includes Content1 and Content2 in that order. Having Points as 
        %INFORMATION ABOUT {contents_str} in 4 points,\n\nSender: {sender}\nRecipient: {recipient} {contents_str}\n\nEmail Text:""",
        
        temperature=0.4,
        max_tokens=contents_length*2,
        top_p=0.8,
        best_of=2,
        frequency_penalty=0.0,
        presence_penalty=0.0)

    return email_final_text.get("choices")[0]['text']


def main_gpt3emailgen():

    st.image('img/image_banner.png')  
    st.markdown('Generate professional sounding emails based on your cheap comments - powered by Artificial Intelligence (OpenAI GPT-3)! Implemented by '
        '[Naggaro](https://www.nagarro.com/en/)'
        )
    st.write('\n') 

    st.subheader('\nWhat is your email all about?\n')
    
    
    with st.expander("SECTION - Email Input", expanded=True):

            input_c1 = st.text_input('Enter email contents down below! (currently 2x seperate topics supported)', 'topic 1')
            input_c2 = st.text_input('', 'topic 2 (optional)')

            email_text = "" 
            col1, col2, col3, space, col4,col5 = st.columns([5, 5, 5, 0.5, 5,5])
            with col1:
                input_sender = st.text_input('Sender Name', '[rephraise]')
            with col2:
                input_recipient = st.text_input('Recipient Name', '[recipient]')
            with col3:
                input_style = st.selectbox('Writing Style',
                                        ('formal', 'motivated', 'concerned', 'disappointed'),
                                        index=0)
            with col5:
                template = st.selectbox('Template Type',  ('Education','Outdoors'))   
                if template=='Education':
                    with open('Templates/Education.html', 'r') as file:  
                        html_string = file.read()
                elif  template=='Outdoors':
                    with open('Templates/Outdoors.html', 'r') as file:  
                        html_string = file.read() 
            with col4:
                st.write("\n")  # add spacing
                st.write("\n")  # add spacing
                if st.button('Generate Email'):
                    with st.spinner():

                        input_contents = []  # let the user input all the data
                        if (input_c1 != "") and (input_c1 != 'topic 1'):
                            input_contents.append(str(input_c1))
                        if (input_c2 != "") and (input_c2 != 'topic 2 (optional)'):
                            input_contents.append(str(input_c2))

                        if (len(input_contents) == 0):  # remind user to provide data
                            st.write('Please fill in some contents for your message!')
                        if (len(input_sender) == 0) or (len(input_recipient) == 0):
                            st.write('Sender and Recipient names can not be empty!')

                        if (len(input_contents) >= 1):  # initiate gpt3 mail gen process
                            if (len(input_sender) != 0) and (len(input_recipient) != 0):
                                email_text = gen_mail_format(input_sender,
                                                            input_recipient,
                                                            input_style,
                                                            input_contents)
        
    with st.container():
        
            st.components.v1.html(html_string,width=700, height=1500, scrolling=True)      
    
    if email_text != "":
        st.write('\n')  # add spacing
        st.subheader('\nHere is the Email Body Content\n')
        with st.expander("SECTION - Email Output", expanded=True):
            st.markdown(email_text)  #output the results

    


if __name__ == '__main__':
    # call main function
    main_gpt3emailgen()
