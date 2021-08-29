from flask import Flask, jsonify, request
from runEatingPlan import runEatingPlan
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
    app.run(debug=True, port=4000)
