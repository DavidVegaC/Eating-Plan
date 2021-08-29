from EatingPlan import EatingPlan
from data import products_tabla,quantity_data
# total_calories = 2500 * 3 Esto es lo que tengo que enviarle como parametro desde el flask

def runEatingPlan (total_calories):
    population_size = 20
    generations = 10
    crossover_probability = 0.8
    mutation_probability = 0.1
    tournament_size = 3
    elitism = True

    problem = EatingPlan(total_calories,population_size,
                generations,crossover_probability,
                mutation_probability,tournament_size,elitism)

    genome=problem.run()
    genome=list(genome)

    EatingPlan1=[]
    for idx in range(len(products_tabla.index)):
        product ={}
        product["name"]=products_tabla['Nombre'][idx]

        if products_tabla['Unidad'][idx] == 'u':
            product["quantity"]=float(round(genome[idx]))
        else:
            product["quantity"]=float(round(genome[idx]*quantity_data[idx].numpy(),1))
        EatingPlan1.append(product)

    #print(EatingPlan1)

    return EatingPlan1