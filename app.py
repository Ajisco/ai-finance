from flask import Flask, request, render_template, session, jsonify
import numpy as np
import pandas as pd
import requests
import openai
import os
import json
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import warnings
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore", category=FutureWarning)





app= Flask(__name__)
app.secret_key = 'poisawoud24e21cjn!Ew@@dsa5'

openai.api_key = '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'
'''
url = "data.csv"
data = pd.read_csv(url)

# filling the mising values
data['Bank'] = data['Bank'].fillna(data['Bank'].mode()[0])
data['BankState'] = data['BankState'].fillna(data['BankState'].mode()[0])
data['NewExist'] = data['NewExist'].fillna(data['NewExist'].mode()[0])
data['xx'] = data['xx'].fillna(data['NewExist'].mode()[0])
data['RevLineCr'] = data['RevLineCr'].replace({'0': 'Y', 'T': 'N'})
data['RevLineCr'] = data['RevLineCr'].fillna(data['RevLineCr'].mode()[0])
data['LowDoc'] = data['LowDoc'].fillna(data['LowDoc'].mode()[0])

# data['LowDoc'] = data['LowDoc'].replace(['S','N'], ['A','N'],['0','N'])
data['LowDoc'] = data['LowDoc'].replace(['S', 'N'], ['A', 'N']).replace('0', 'N')
data['LowDoc'] = data['LowDoc'].replace('A','N')

# dropping the non-important values
cols_to_drop = ['Name','City','State','ChgOffDate','DisbursementDate','BalanceGross','Bank','BankState']
data.drop(cols_to_drop, axis=1, inplace = True)

# Label Encoding is to convert the categorical dataset into numerical
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cols_to_encode = ['RevLineCr', 'LowDoc', 'MIS_Status']

for col in cols_to_encode:
    data[col] = le.fit_transform(data[col])

#scale numerical features for logistic model
features = data.drop(columns=['Default']).columns
target = 'Default'

# define standard scaler
scaler = StandardScaler()

# transform data
data[features] = scaler.fit_transform(data[features])


new_cols = ['MIS_Status', 'ChgOffPrinGr', 'Term', 'daysterm', 'SBA_Appv',
       'ApprovalDate', 'xx', 'LoanNr_ChkDgt', 'GrAppv', 'ApprovalFY',target]

new_data = data[new_cols]

#split train data into train and validation set
X_train, X_test, y_train, y_test = train_test_split(data[features],
                                                    data[target].to_frame(),
                                                    test_size=0.3,
                                                    random_state=1234)

new_rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
new_rf_classifier.fit(X_train, y_train)

'''
def chatGPT(text):
  url = "https://api.openai.com/v1/completions"
  headers = {
  "Content-Type": "application/json",
  "Authorization": "Bearer YOUR-API-KEY",
  }
  data = {
  "model": "text-davinci-003",
  "prompt": text,
  "max_tokens": 4000,
  "temperature": 0.6,
  }
  response = requests.post(url, headers=headers, json=data)
  output = response.json()["choices"][0]["text"]

  return print(output)







def get_response(prompt, model="gpt-3.5-turbo"):

  messages = [{"role": "user", "content": prompt}]

  messages=[
      {"role": "system", "content": "You are a nice loan acceptance prediction and assistant for small business enterprises"},
      {"role": "user", "content": prompt},
  ]

  response = openai.ChatCompletion.create(

  model=model,

  messages=messages,

  temperature=0,

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
  if prediction == 1: #Yes
    add_text = "Again Congrats on your loan application that looks like it would be approved"
  elif prediction == 0: #No
    add_text = 'Sorry about your loan application that might not be approved'

  else:
    add_text = ""

  final_previous_conv += add_text

  new_prompt = "Hi, this is my question: "+question+" .Answer based on our previous conversion which is: "+final_previous_conv+". Go straight to the point and make the reply strictly less than 800 characters."

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


'''
@app.route('/form_predict', methods=["GET", "POST"])
def form_predict():
    return render_template('form_predict.html')
'''


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




@app.route('/form_predict', methods=["GET", "POST"])
def index():
    '''
    name = request.sign_in['name'].capitalize()
    country= request.sign_in['country']
    stud_hr= request.form['stud_hr']
    employed= request.form['employed']
    h_disab= request.form['h_disab']
    ment_cond= request.form['ment_cond']
    social_hr= request.form['social_hr']
    fit_hr= request.form['fit_hr']
    wind= request.form['wind']
    dry_mouth= request.form['dry_mouth']
    positive= request.form['positive']
    breath_diff= request.form['breath_diff']
    initiate= request.form['initiate']
    tremb= request.form['tremb']
    worry= request.form['worry']
    look_fwd= request.form['look_fwd']
    down= request.form['down']
    enthus= request.form['enthus']
    life_mean= request.form['life_mean']
    scared= request.form['scared']
    arr = pd.DataFrame((np.array([[age,stud_hr,employed,h_disab,ment_cond,social_hr,fit_hr,wind,dry_mouth,
                positive,breath_diff,initiate,tremb,worry,look_fwd,down,enthus,
                life_mean,scared]])
        ), columns='X_train'.columns)
    pred= model.predict(arr)
    '''

    pred = 1
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
    business_response = jsonify({"bot_business_response": bot_business_response })

    session["bot_business_response"] = business_response
    session["bot_business_prompt"] = bot_business_prompt

    return render_template('chat_business.html',name = name,
                           country = country, bot_business_response = bot_business_response)



@app.route('/further_business_chat', methods=["GET", "POST"])
def further_business_chat():

  bot_business_response = session.get("bot_business_response",None)
  bot_business_prompt = session.get("bot_business_prompt",None)

  if request.method == 'POST':
      business_question = request.form['question']

      business_response = get_further_response(prediction = "", question = business_question,
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



    finance_response = jsonify({"bot_finance_response": bot_finance_response })

    session["bot_finance_response"] = finance_response
    session["bot_finance_prompt"] = bot_finance_prompt

    return render_template('chat_business.html',name = name,
                           country = country, bot_finance_response = bot_finance_response)



@app.route('/further_finance_chat', methods=["GET", "POST"])
def further_finance_chat():

  bot_finance_response = session.get("bot_finance_response",None)
  bot_finance_prompt = session.get("bot_finance_prompt",None)

  if request.method == 'POST':
      finance_question = request.form['question']

      finance_response = get_further_response(prediction = "", question = finance_question,
                                      prev_prompt = bot_finance_prompt, prev_response = bot_finance_response)

  session["bot_business_response"] = finance_response
  session["bot_business_prompt"] = finance_question


  return jsonify({"response": finance_response })

if __name__ == '__main__':
    app.run(debug= True, use_reloader=False)