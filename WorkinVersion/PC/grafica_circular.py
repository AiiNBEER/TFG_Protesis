import matplotlib.pyplot as plt

# Data from the provided table
labels = ['Visión', 'Audición', 'Comunicación', 'Aprendizaje y desarrollo de tareas', 
          'Movilidad', 'Autocuidado', 'Vida doméstica', 'Relaciones personales']

# Summing the total number of each type of disability for both sexes
sizes = [sum([84579, 26250, 58329]), 
         sum([86315, 25564, 60751]), 
         sum([221069, 73368, 147701]), 
         sum([225512, 70992, 154520]), 
         sum([308351, 100233, 208118]), 
         sum([317011, 104387, 212624]), 
         sum([311043, 103506, 207537]), 
         sum([142482, 48568, 93914])]

# Plotting the pie chart
fig, ax = plt.subplots()

ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title("Distribución de tipos de discapacidad")
plt.show()
