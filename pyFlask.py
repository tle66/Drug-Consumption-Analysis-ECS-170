from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from joblib import load

app = Flask(__name__)
model = load('./gatewayModel.joblib')
demographicModels=[]
PersonalityModels=[]
for i in range(1, 9):
    PersonalityModels.append(load("./personalityModels/PersonalityModel"+str(i)+".joblib"))
    demographicModels.append(load("./demographicModels/demographicModel"+str(i)+".joblib"))

# Used to change user input to actual value used in model
def getNScoreVal(num):
            #12, 13, 14, 15, 16...
    lst = [
        -3.46436,-3.15735,-2.75696,-2.52197,-2.42317,
        -2.34360,-2.21844,-2.05048,-1.86962,-1.69163,
        -1.55078,-1.43907,-1.32828,-1.19430,-1.05308,
        -0.92104,-0.79151,-0.67825,-0.58016,-0.46725,
        -0.34799,-0.24649,-0.14882,-0.05188,0.04257,
        0.13606,0.22393,0.31287,0.41667,0.52135,
        0.62967,0.73545,0.82562,0.91093,1.02119,
        1.13281,1.23461,1.37297,1.49158,1.60383,
        1.72012,1.83990,1.98437,2.12700,2.28554,
        2.46262,2.61139,2.82196,3.27393
    ]
    return lst[num - 12]

def getEScoreVal(num):
    lst = [
        -3.27393,-3.27393,-3.00537,-2.72827,-2.53830, 
        -2.44904, -2.32338, -2.21069, -2.11437, -2.03972, 
        -1.92173, -1.76250, -1.63340, -1.50796, -1.37639, 
        -1.23177,-1.09207,-0.94779,-0.80615,-0.69509,
        -0.57545,-0.43999,-0.30033,-0.15487,0.00332,
        0.16767,0.32197,0.47617,0.63779,0.80523,
        0.96248,1.11406,1.28610,1.45421,1.58487,
        1.74091,1.93886,2.12700,2.32338,2.57309,
        2.85950,3.00537,3.00537,3.27393
    ]
    return lst[num-16]

def getOScoreVal(num):
    lst = [
        -3.27393,-3.27393,-2.85950, -2.85950,-2.63199, 
        -2.39883, -2.21069, -2.09015, -1.97495, -1.82919, 
        -1.68062, -1.55521, -1.42424, -1.27553,-1.11902,
        -0.97631,-0.84732,-0.71727,-0.58331,-0.45174,
        -0.31776,-0.17779,-0.01928,0.14143,0.29338,
        0.44585,0.58331,0.72330,0.88309,1.06238,
        1.24033,1.43533,1.65653,1.88511,2.15324,
        2.44904,2.90161
    ]
    return lst[num-24]

def getAScoreVal(num):
    lst = [
        -3.46436,-3.46436,-3.46436,-3.46436,-3.15735,
        -3.15735,-3.00537,-3.00537,-3.00537,-3.00537,
        -3.00537,-2.90161,-2.78793,-2.70172,-2.53830,
        -2.35413,-2.21844,-2.07848,-1.92595,-1.77200,
        -1.62090,-1.47955,-1.34289,-1.21213,-1.07533,
        -0.91699,-0.76096,-0.60633,-0.45321,-0.30172,
        -0.15487,-0.01729,0.13136,0.28783,0.43852,
        0.59042,0.76096,0.94156,1.11406,1.2861,
        1.45039,1.61108,1.81866,2.03972,2.23427,
        2.46262,2.75696,3.15735,3.46436
    ]
    return lst[num-12]

def getCScoreVal(num):
    lst = [
        -3.46436,-3.46436,-3.15735,-2.90161,-2.72827,
        -2.57309,-2.42317,-2.30408,-2.18109,-2.04506,
        -1.92173,-1.78169,-1.64101,-1.5184,-1.38502,
        -1.25773,-1.13788,-1.0145,-0.89891,-0.78155,
        -0.65253,-0.52745,-0.40581,-0.27607,-0.14277,
        -0.00665,0.12331,0.25953,0.41594,0.58489,
        0.7583,0.93949,1.13407,1.30612,1.46191,
        1.63088,1.81175,2.04506,2.33337,2.63199,
        3.00537,3.00537,3.46436
    ]
    return lst[num-17]

def getIScoreVal(num):
    lst = [
        -2.55524,-1.37983,-0.71126,-0.21712,0.19268,
        0.52975,0.88113,1.29221,1.86203,2.90161
    ]
    return lst[num-1]

def getSScoreVal(num):
    lst = [
        -2.07848,-1.54858,-1.18084,-0.84637,-0.52593,
        -0.21575,0.07987,0.40148,0.76540,1.22470,
        1.92173
    ]
    return lst[num-1]

@app.route('/')
def home():
    return render_template('homePage.html')

@app.route('/page1.html')
def page1():
    return render_template('page1.html')

@app.route('/page2.html')
def page2():
    return render_template('page2.html')

@app.route('/page3.html')
def page3():
    return render_template('page3.html')

@app.route('/page1.html/predictPersonality', methods=['POST'])
def predictPersonality():
    nScore = request.form['nss']
    eScore = request.form['ess']
    oScore = request.form['oss']
    aScore = request.form['ass']
    cScore = request.form['css']
    iScore = request.form['iss']
    sScore = request.form['sss']
    lst= [[ getNScoreVal(int(nScore)), getEScoreVal(int(eScore)),
           getOScoreVal(int(oScore)), getAScoreVal(int(aScore)),
           getCScoreVal(int(cScore)), getIScoreVal(int(iScore)),
           getSScoreVal(int(sScore))
    ]]
    df = pd.DataFrame(lst, columns=["Nscore", "Escore", "Oscore", "Ascore", "Cscore", "Impulsive", "SS"])
    messages=[]
    for model in PersonalityModels:
        messages.append(model.predict(df))

    for idx in range(len(messages)):
        if messages[idx][0] == 0:
            messages[idx] = "Never Have Used"
        else:
            messages[idx] = "Have Used Before"

    selected_substances = ['Alcohol', 'Amphet', 'Cannabis', 'Coke', 'Ecstasy', 'LSD', 'Meth', 'Mushrooms']
    for index, substance in enumerate(selected_substances):
        messages[index] = substance + " usage: " + messages[index]

    return render_template("page1.html", alcPredText=messages[0], ampPredText=messages[1], canPredText=messages[2],
                           cokPredText=messages[3], ecsPredText=messages[4], lsdPredText=messages[5],
                           metPredText=messages[6],
                           musPredText=messages[7])

@app.route('/page2.html/predictDemographic', methods=['POST'])
def predictDemographic():

    age = request.form['age']
    gender = request.form['gender']
    education = request.form['education']
    country = request.form['country']
    eth = request.form['ethnicity']
    nScore = request.form['nss']
    eScore = request.form['ess']
    oScore = request.form['oss']
    aScore = request.form['ass']
    cScore = request.form['css']
    iScore = request.form['iss']
    sScore = request.form['sss']
    lst = [[float(age), float(gender), float(education), float(country), float(eth),
           getNScoreVal(int(nScore)), getEScoreVal(int(eScore)),
           getOScoreVal(int(oScore)), getAScoreVal(int(aScore)),
           getCScoreVal(int(cScore)), getIScoreVal(int(iScore)),
           getSScoreVal(int(sScore))
    ]]

    df = pd.DataFrame(lst, columns=["Age", "Gender", "Education", "Country", "Ethnicity", "Nscore", "Escore", "Oscore", "Ascore", "Cscore", "Impulsive", "SS"])
    messages = []
    for model in demographicModels:
        messages.append(model.predict(df))

    for idx in range(len(messages)):
        if messages[idx][0] == 0:
            messages[idx] = 'Never Have Used'
        else:
            messages[idx] = 'Have Used Before'

    selected_substances = ['Alcohol', 'Amphet', 'Cannabis', 'Coke', 'Ecstasy', 'LSD', 'Meth', 'Mushrooms']
    for index, substance in enumerate(selected_substances):
        messages[index] = substance + " usage: " + messages[index]

    return render_template("page2.html", alcPredText=messages[0], ampPredText=messages[1],canPredText=messages[2],
    cokPredText=messages[3],ecsPredText=messages[4],lsdPredText=messages[5],metPredText=messages[6],
    musPredText=messages[7])

@app.route('/page3.html/predictGateway', methods=['POST'])
def predictGateway():
    '''
    For rendering results on HTML GUI
    '''
    command = request.form['alcohol']
    command2 = request.form['cannabis']
    lst = [[int(command), int(command2)]]
    df = pd.DataFrame(lst, columns=['Alcohol', 'Cannabis'])
    prediction = model.predict(df)
    if prediction[0] == 2:
        predicted_res = 'Recent User (within the last year)'
    elif prediction[0] == 1:
        predicted_res = 'Seldom User (within the last decade)'
    else:
        predicted_res = 'Never Have Used'
    return render_template('page3.html', prediction_text='Predicted hard drug use based on your consumption of alcohol/cannabis: {}'.format(predicted_res))

if __name__ == "__main__":

    app.run()

