from shiny import App, render, ui, reactive
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

x_encoder = pickle.load(open("encoder", 'rb'))
polynomial = pickle.load(open("poly", 'rb'))
model = pickle.load(open("british_voter", 'rb'))

app_ui = ui.page_fluid(
    "Input a British voter. This voter is a:", 
    ui.input_select("ethnicity", "A", {"White British": "White British", "Any other white background": "other White",\
                                      "Asian": "Asian", "Black":"Black", "Mixed Race": "Mixed Race", "Other ethnic group":"other"}),
    ui.input_select("religion", "", {"Anglican": "Anglican", "non-Anglican Protestant": "non-Anglican Protestant",\
                                    "Catholic": "Catholic", "Orthodox Christian":"Orthodox Christian", "Islam":"Muslim",\
                                     "Hinduism":"Hindu", "Judaism":"Jewish", "Buddhism":"Buddhist", "Sikhism":"Sikh",\
                                    "No religion":"non-religious", "Other":"other"}), 
    ui.input_select("sexuality", "", {"Straight": "straight", "LGB+": "LGB+"}),
    ui.input_select("gender", "", {"Male": "man", "Female": "woman"}), 
    ui.input_select("age_cat", "", {"18-29":"between 18 and 29", "30-44":"between 30 and 44", "45-65":"between 45 and 64", "65+": "65 or older"}), 
    ui.input_select("married", "They are", {"Married": "married", "Unmarried": "unmarried"}), 
    ui.input_select("children", "", {"Yes": "have children", "No": "have no children"}), 
    ui.input_select("education", "and acheived", {"No qualifications": "no formal qualifications",\
                                                  "GCSE or equivalent": "GCSE or equivalent qualifications",\
                                                  "A-level or equivalent": "A-level or equivalent qualifications",\
                                                  "Technical or professional qualification": "a technical or professional qualification",\
                                                  "University degree":"a university degree",\
                                                  "Post-graduate degree": "a post-graduate degree"}), 
    ui.input_select("work", "They are", {"full time": " working full time", "part time":"working part time",\
                                        "student":"a full time student", "retired":"retired", "unemployed":"unemployed", "other":"other"}),
    ui.input_select("income", "and earn", {"under £5,000 per year":"under £5,000 per year", "£5,000 to £9,999 per year":"£5,000 to £9,999 per year",\
                                          "£10,000 to £14,999 per year": "£10,000 to £14,999 per year",\
                                          "£15,000 to £19,999 per year": "£15,000 to £19,999 per year",\
                                          "£20,000 to £24,999 per year": "£20,000 to £24,999 per year",\
                                          "£25,000 to £29,999 per year": "£25,000 to £29,999 per year",\
                                          "£30,000 to £34,999 per year": "£30,000 to £34,999 per year",\
                                          "£35,000 to £39,999 per year": "£35,000 to £39,999 per year",\
                                          "£40,000 to £44,999 per year": "£40,000 to £44,999 per year",\
                                          "£45,000 to £49,999 per year": "£45,000 to £49,999 per year",\
                                          "£50,000 to £59,999 per year": "£50,000 to £59,999 per year",\
                                          "£60,000 to £69,999 per year": "£60,000 to £69,999 per year",\
                                          "£70,000 to £99,999 per year": "£70,000 to £99,999 per year",\
                                          "£100,000 to £149,999 per year": "£100,000 to £149,999 per year",\
                                          "£150,000 and over": "more than £150,000 per year"}), 
    ui.input_select("housing", "They live in a home they", {"own outright": "own outright", "own with mortgage": "own with mortgage",\
                                                           "private rental": "rent privately", "social housing": "rent socially",\
                                                           "other": "other"}), 
    ui.input_select("rural_urban", "in", {"urban": "an urban area", "mostly urban": "a mostly urban area",\
                                         "mostly rural": "a mostly rural area", "rural": "a rural area"}), 
    ui.input_select("region", "in", {"South East":"the South East", "London": "London", "East of England": "the East of England",\
                                    "South West": "the South West", "West Midlands": "the West Midlands", "East Midlands": "the East Midlands",\
                                    "North West": "the North West", "North East": "the North East", "Yorkshire and The Humber": "Yorkshire and The Humber",\
                                     "Wales": "Wales", "Scotland": "Scotland"}),
    ui.output_plot("plot"),
)


def server(input, output, session): 
    
    @reactive.Calc
    def compute_probs():
        voter_dict = {"ethnicity":[input.ethnicity()], "religion":[input.religion()], "gender":[input.gender()], "housing":[input.housing()],\
                      "work":[input.work()], "sexuality":[input.sexuality()], "age_cat":[input.age_cat()], "married":[input.married()],\
                      "children":[input.children()], "education":[input.education()], "income":[input.income()], "region":[input.region()],\
                      "rural_urban":[input.rural_urban()]}
        x_transform = x_encoder.transform(pd.DataFrame.from_dict(voter_dict))
        x_transform = polynomial.transform(x_transform)
        probs = model.predict_proba(x_transform)[0]
        prob_dict = {"Brexit Party/Reform UK": probs[0],\
                "Conservative": probs[1],\
                "Green Party": probs[2],\
                "Labour": probs[3],\
                "Liberal Democrat": probs[4],\
                "Scottish National Party (SNP)": probs[5]}

        return prob_dict
    
    colors = {"Brexit Party/Reform UK":"purple",\
              "Conservative":"blue",\
              "Green Party":"green",\
              "Labour":"red",\
              "Liberal Democrat":"orange",\
              "Scottish National Party (SNP)":"yellow"}
    @output 
    @render.plot    
    def plot(): 
        names = list(compute_probs().keys())
        values = list(compute_probs().values())
        fig, ax = plt.subplots()
        ax.bar(range(6), values, tick_label=names, color = [colors[i] for i in names])
        return fig



        
app = App(app_ui, server)
