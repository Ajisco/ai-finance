from flask import Flask, request, render_template, session, jsonify
import numpy as np
import pandas as pd
import requests
import openai
import os
import json
import time
import joblib
from openai import OpenAI

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score
# from sklearn.ensemble import RandomForestClassifier
#
#dataset = pd.read_csv("data.csv")

# dataset.drop(['loan_id'], axis = 1, inplace = True)

# dataset.columns = dataset.columns.str.strip()

# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()

# dataset['self_employed'] = le.fit_transform(dataset['self_employed'])
# dataset['loan_status'] = le.fit_transform(dataset['loan_status'])
# dataset['education'] = le.fit_transform(dataset['education'])



# # Classification Modeling

# # Split dataset
# X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]

# # Create train and test splits
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# clf = RandomForestClassifier(
#                    max_depth=7, min_samples_split=5, min_samples_leaf=5, random_state=42)

# # Train the model
# clf.fit(X_train, y_train)


# # Save the model to a file
# joblib.dump(clf, 'random_forest_model.pkl')

# print("Model saved successfully.")



app= Flask(__name__)
app.secret_key = 'poisawoud24e21cjn!Ew@@dsa5'

# Load the model from the file
loaded_model = joblib.load('random_forest_model.pkl')


print("Model loaded successfully.")

columns = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
       'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
       'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']




# Instantiate the OpenAI client
client = OpenAI(api_key='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') 

def chatGPT(text):
    completion = client.completions.create(
        model="text-davinci-003",
        prompt=text,
        max_tokens=4000,
        temperature=0.6
    )
    return print(completion.choices[0].text)

def get_response(prompt, model="gpt-3.5-turbo"):
    messages = [
        {"role": "system", "content": "You are a nice loan acceptance prediction and assistant for small business enterprises"},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message["content"]







def get_predict_message(country):
  format = '''[
    {
    "myCounntry": {
    "organizationName": "",
    "link": ""
    },
    "otherCountry": {
      "organizationName":"",
    "link": "",
    "Country": ""
    }
    },

    {
    "myCounntry": {
    "organizationName": "",
    "link": ""
    },
    "otherCountry": {
      "organizationName":"",
    "link": "",
    "Country": ""
    }
    }
    ]
  '''
  prompt = "Hi, my country is "+country+ ". Kindly create a comprehensive list of places I can look out for to get a good loan for my small business establishment both in my country and other countries.Make sure you give the countries of the other countries!. Give the answer strictly in this format: "+format+". Thanks."

  prompt_response = get_response(prompt)

  return prompt, prompt_response




def get_further_response(prediction,question, prev_prompt, prev_response):
  old = str(prev_prompt)+str(prev_response)
  previous_conv = ""
  rev_old = old[::-1]
  for char in rev_old:
    if len(previous_conv) < 2500:
      previous_conv+char
  final_previous_conv = previous_conv[::-1]
  if prediction == 0: #Yes
    add_text = "again congrats on your approved loan"
  elif prediction == 1: #No
    add_text = 'again sorry about the unapproved loan'

  else:
    add_text = ""

  final_previous_conv += add_text

  new_prompt = "Question: " + question + " | Previous Context: " + final_previous_conv + " | Instruction: Provide a concise, direct answer within 800 characters."

  further_response = get_response(new_prompt)

  return new_prompt, further_response







def get_business_idea(country,country_interest, capital_loan, amount, domain_interest, loan_pay_month):
  format = '''
  [
    {
        "Business_Idea": "",
        "sector": "",
        "link": ""
    },
   {
        "Business Idea": "",
        "sector": "",
        "link": ""
    }
]
'''
  if capital_loan == 'capital':
    prompt = "Hi, I'm from "+country+". Kindly help curate few nice business ideas, the domain sector of the business and like to learn more on the business, considering that I have a capital of "+amount+" US Dollars. My domain of business interest is "+domain_interest+" and the country where I want to have my business is "+country_interest+". Give the answer strictly in this format: "+format+" Thanks."

  elif capital_loan == 'loan':
    prompt = "Hi, I'm from "+country+". Kindly help curate few nice business ideas, the domain sector of the business and like to learn more on the business, considering that I got a loan of "+amount+" US Dollars and I am meant to pay back in "+loan_pay_month+" months time. My domain of business interest is "+domain_interest+" and the country where I want to have my business is "+country_interest+". Give the answer strictly in this format: "+format+" Thanks."

  idea_response = get_response(prompt)

  return prompt, idea_response







def get_financial_advice(country,country_interest,description, capital_loan, amount, domain_interest, loan_pay_month):
  format = '''
  {
        "financial_breakdown": "",
        "link": ""
    }
'''
  if capital_loan == 'capital':
    prompt = "Hi, I'm from "+country+". Kindly help curate a comprehensive financial breakdown with link to read more on it, for how I would manage my business considering that I have a capital of "+amount+" US Dollars. My domain of business interest is "+domain_interest+",the description is: "+description+" and the country where I want to have my business is "+country_interest+". Make your answer strictly in this format: "+format+" ."
  elif capital_loan == 'loan':
    prompt = "Hi, I'm from "+country+". Kindly help curate a comprehensive financial breakdown with link to read more on it, for how I would manage my business considering that I got a loan of "+amount+" US Dollars and I am meant to pay back in "+loan_pay_month+" months time. My domain of business interest is "+domain_interest+",the description is: "+description+" and the country where I want to have my business is "+country_interest+". Make your answer strictly in this format: "+format+" ."

  advice_response = get_response(prompt)

  return prompt, advice_response







model= None

@app.route('/', methods=["GET", "POST"])
def main():
    return render_template('index.html')



@app.route('/form_predict', methods=["GET", "POST"])
def form_predict():
    return render_template('form_predict.html')



@app.route('/form_business_idea', methods=["GET", "POST"])
def form_business_idea():
    return render_template('form_business_idea.html')



@app.route('/sign_in', methods=["GET", "POST"])
def sign_in():
    return render_template('sign_in.html')



@app.route('/services', methods=["GET", "POST"])
def services():
    return render_template('services.html')



@app.route('/form_financial_advice', methods=["GET", "POST"])
def form_financial_advice():
    return render_template('form_financial_advice.html')



@app.route('/next_session', methods=["GET", "POST"])
def next_session():
  name = request.form['name'].capitalize()
  country= request.form['country']
  session["name"]=name
  session["country"]= country
  return render_template('services.html', country=country ,name = name)




@app.route('/chat_predict', methods=["GET", "POST"])
def chat_predict():
    depend= request.form['depend']
    education= request.form['education']
    employment= request.form['employment']
    income= request.form['income']
    loan_amount= request.form['loan_amount']
    loan_term= request.form['loan_term']
    score= request.form['score']
    resident= request.form['resident']
    commercial= request.form['commercial']
    luxury= request.form['luxury']
    bank= request.form['bank']
    arr = pd.DataFrame((np.array([[depend,education,employment,income,loan_amount,
                                   loan_term,score,resident,commercial,
                luxury,bank]])
        ), columns=columns)
    pred= int(loaded_model.predict(arr)[0])


    country = session.get("country",None)
    name = session.get("name",None)

    bot_predict_prompt, bot_predict_response = get_predict_message(country)

    #bot_predict_response =  jsonify({"bot_predict_response": bot_predict_response})

    bot_predict_response = json.loads(bot_predict_response)

    session["pred"]=pred
    session["bot_predict_response"] = bot_predict_response
    session["bot_predict_prompt"] = bot_predict_prompt

    return render_template('chat_predict.html', pred=pred ,name = name,
                            country = country, bot_predict_response = bot_predict_response)



@app.route('/further_predict_chat', methods=["GET", "POST"])
def further_predict_chat():

  pred = session.get("pred",None)
  bot_predict_prompt = session.get("bot_predict_prompt",None)
  bot_predict_response = session.get("bot_predict_response",None)

  if request.method == 'POST':
      predict_question = request.form['question']

      predict_prompt, predict_response = get_further_response(prediction = pred, question = predict_question,
                                      prev_prompt = bot_predict_prompt, prev_response = bot_predict_response)

  session["bot_predict_response"] = predict_response
  session["bot_predict_prompt"] = predict_question


  return jsonify({"response": predict_response })


@app.route('/business_idea', methods=["GET", "POST"])
def business_idea():
    country_interest = request.form['country_interest'].capitalize()
    capital_loan= request.form['capital_loan']
    amount= request.form['amount']
    domain_interest= request.form['domain_interest']
    loan_pay_month= request.form['loan_pay_month']

    country = session.get("country",None)
    name = session.get("name",None)

    bot_business_prompt , bot_business_response  = get_business_idea(country = country,
                                                                     country_interest=country_interest,
                                                                     capital_loan=capital_loan,
                                                                     amount=amount,
                                                                     domain_interest=domain_interest,
                                                                     loan_pay_month=loan_pay_month)
    
    bot_business_response = json.loads(bot_business_response) 

    session["bot_business_response"] = bot_business_response
    session["bot_business_prompt"] = bot_business_prompt

    return render_template('chat_business.html',name = name,
                           country = country, bot_business_response = bot_business_response)



@app.route('/further_business_chat', methods=["GET", "POST"])
def further_business_chat():

  bot_business_response = session.get("bot_business_response",None)
  bot_business_prompt = session.get("bot_business_prompt",None)

  if request.method == 'POST':
      business_question = request.form['question']

      business_prompt,  business_response = get_further_response(prediction = "", question = business_question,
                                      prev_prompt = bot_business_prompt, prev_response = bot_business_response)

  session["bot_business_response"] = business_response
  session["bot_business_prompt"] = business_question


  return jsonify({"response": business_response })




@app.route('/financial_advice', methods=["GET", "POST"])
def financial_advice():
    country_interest = request.form['country_interest'].capitalize()
    capital_loan= request.form['capital_loan']
    description= request.form['description']
    amount= request.form['amount']
    domain_interest= request.form['domain_interest']
    loan_pay_month= request.form['loan_pay_month']

    country = session.get("country",None)
    name = session.get("name",None)

    bot_finance_prompt , bot_finance_response = get_financial_advice(country = country,
                                                                     country_interest=country_interest,
                                                                     description = description,
                                                                     capital_loan = capital_loan,
                                                                     amount=amount,
                                                                     domain_interest=domain_interest,
                                                                     loan_pay_month=loan_pay_month)



    bot_finance_response = json.loads(bot_finance_response) 
    
    session["bot_finance_response"] = bot_finance_response
    session["bot_finance_prompt"] = bot_finance_prompt

    return render_template('chat_finance.html',name = name,
                           country = country, bot_finance_response = bot_finance_response)



@app.route('/further_finance_chat', methods=["GET", "POST"])
def further_finance_chat():

  bot_finance_response = session.get("bot_finance_response",None)
  bot_finance_prompt = session.get("bot_finance_prompt",None)

  if request.method == 'POST':
      finance_question = request.form['question']

      finance_prompt, finance_response = get_further_response(prediction = "", question = finance_question,
                                      prev_prompt = bot_finance_prompt, prev_response = bot_finance_response)

  session["bot_business_response"] = finance_response
  session["bot_business_prompt"] = finance_question


  return jsonify({"response": finance_response })

if __name__ == '__main__':
    app.run(debug= True, use_reloader=False)
