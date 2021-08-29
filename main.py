from flask import Flask, jsonify, request
from runEatingPlan import runEatingPlan
from decouple import config as config_decouple

app = Flask(__name__)

# Testing Route
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'response': 'pong!'})

@app.route('/planAlimenticio/<int:total_calories>')
def getProduct(total_calories):
    response = runEatingPlan (total_calories)
    return jsonify(response)


if __name__ == '__main__':
    if config_decouple('PRODUCTION', default=False):
        app.run(debug=False, port=4000)
    else: 
        app.run(debug=True, port=4000)

