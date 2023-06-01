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
#################################################################################################
##################################
###############
#######

def gen_mail_contents(email_contents,sender,recipient,style):     

        openaiq = OpenAI(temperature=0.7,openai_api_key="4461d4ebc79a45bca18557145962a4f3",deployment_id="EmailGeneratorDemo02")                                         
        map_prompt = """Below is a section of a information about {prospect}
       Write a concise summary about {prospect}. If the information is not about {prospect}, exclude it from your summary.{text}
% CONCISE SUMMARY:"""
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "prospect"])

        combine_prompt = """
Your goal is to write the email content for {Reason} by {company} to {prospect}.
A good campaign email combines information about the reason of the campaign.Length of email should be less than 50 words.
% INFORMATION ABOUT {company}:
{company_information}

% INFORMATION ABOUT {prospect}:
{text}

% INCLUDE THE FOLLOWING PIECES IN YOUR RESPONSE:

- Start the email with 'We'
- Keep the email in {style} tone 
- Start with {Reason} of the email

- A 1-2 sentence description about {company}, be brief
- End your content with a call-to-action such as asking them to set up time to talk more


% YOUR RESPONSE:
"""
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["Reason", "company", "prospect", \
                                                                         "text","company_information","style"])

        chain = load_summarize_chain(openaiq,
                             chain_type="map_reduce",
                             map_prompt=map_prompt_template,
                             combine_prompt=combine_prompt_template,
                             verbose=True
                            )
        text_splitter = CharacterTextSplitter()

        ############################ Information about the Prospect
        with open("./BlackBaud.txt") as f:
            state_of_the_union = f.read()
            texts = text_splitter.split_text(state_of_the_union)
       
        docs = [Document(page_content=t) for t in texts[:3]]

        ############################ Information about the Sender
        Filename="./"+sender+".txt"

        with open(Filename) as f:
            state_of_the_Naggaro = f.read()
        ############################### Getting into chain    

        output = chain({ "input_documents": docs,##These Docs are basically data about the Targeted customer
                "company": sender, \
                "company_information" :state_of_the_Naggaro ,\
                  "Reason" : email_contents[0], \
                "prospect" : recipient,\
                "style" : style
               })
        langout=output['output_text']
        print(langout)       
        
        return langout
###################################################################
################################
###################

def gen_mail_format(sender, recipient, style, email_contents,input_target):
    email_contents = gen_mail_contents(email_contents,sender,recipient,style)
    print(email_contents)
    return email_contents
#########################################################
#####################################
####################

def generate_Variable_content(Variable):
    openaiq = OpenAI(temperature=0.7,openai_api_key="4461d4ebc79a45bca18557145962a4f3",deployment_id="EmailGeneratorDemo02")                                         
    map_prompt = """Below is a section of a information about {prospect}
       Write a concise summary about {prospect}. If the information is not about {prospect}, exclude it from your summary.{text}
% CONCISE SUMMARY:"""
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "prospect"])

    combine_prompt = """
Your goal is to write the one line for {Reason} by {company} to {prospect}.
A good campaign email combines information about the reason of the campaign.Length of email should be less than 50 words.
% INFORMATION ABOUT {company}:
{company_information}

% INFORMATION ABOUT {prospect}:
{text}

% INCLUDE THE FOLLOWING PIECES IN YOUR RESPONSE:

- Start the email with 'We'
- Keep the email in {style} tone 
- Start with {Reason} of the email

- A 1-2 sentence description about {company}, be brief
- End your content with a call-to-action such as asking them to set up time to talk more


% YOUR RESPONSE:
"""
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["Reason", "company", "prospect", \
                                                                         "text","company_information","style"])

    chain = load_summarize_chain(openaiq,
                             chain_type="map_reduce",
                             map_prompt=map_prompt_template,
                             combine_prompt=combine_prompt_template,
                             verbose=True
                            )
    text_splitter = CharacterTextSplitter()

        ############################ Information about the Prospect
    with open("./BlackBaud.txt") as f:
            state_of_the_union = f.read()
            texts = text_splitter.split_text(state_of_the_union)
       
    docs = [Document(page_content=t) for t in texts[:3]] 
    output = chain({ "input_documents": docs,##These Docs are basically data about the Targeted customer
                "company": sender, \
                "company_information" :state_of_the_Naggaro ,\
                  "Reason" : email_contents[0], \
                "prospect" : recipient,\
                "style" : style
               })
    langout=output['output_text']
    print(langout)       
        
    return langout


    pass







##########################################################
#######################################
#####################

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
            col1, col2, col3, space,col5,col6,col4 = st.columns([5, 5, 5, 0.5, 5,5,5])
            with col1:
                input_sender = st.text_input('Sender Name', '[rephraise]')
            with col2:
                input_recipient = st.text_input('Recipient Name', '[recipient]')
            with col3:
                input_style = st.selectbox('Writing Style',
                                        ('formal', 'motivated', 'concerned', 'disappointed'),
                                        index=0)
            with col6:
                input_target=st.selectbox('Customer target',('High Value','Low Value'))
            with col5:
                template = st.selectbox('Template Type',  ('Template1','Template2'))   
                if template!='':
                    variable="Templates"+'\\'+template+".html"
                    with open(variable, 'r') as file:  
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
                                                            input_contents,input_target)
        
          
    
    if email_text != "":
        st.write('\n')  # add spacing
        with st.container():

            html_string=html_string.replace("DATA1 ",email_text)
            html_string=html_string.replace("Variable1",input_c1)
            print (html_string)
                    
            st.components.v1.html(html_string,width=700, height=1500, scrolling=True)
            Variable2 = st.text_input('Variable2', 'Enter a Heading')
            Variable3 = st.text_input('Variable3', 'Enter a Heading')

      #  st.subheader('\nHere is the Email Body Content\n')
      #  with st.expander("SECTION - Email Output", expanded=True):
      #      st.markdown(email_text)  #output the results

    
if __name__ == '__main__':
    # call main function
    main_gpt3emailgen()
