// Obtener el contexto del canvas
    const ctx = document.getElementById('myPieChart').getContext('2d');

    // Crear la gráfica
    new Chart(ctx, {
      type: 'pie', // tipo: pastel
      data: {
//        labels: ['Por tránsito', 'Por velocidad radial'],
        datasets: [{
          label: 'Método de descubrimiento',
          data: [95, 5], // ← los valores (porcentaje o cantidad)
          backgroundColor: [
            'rgba(75, 192, 192, 0.7)',
            'rgba(255, 205, 86, 0.7)',
            'rgba(255, 99, 132, 0.7)',
            'rgba(54, 162, 235, 0.7)'
          ],
          borderColor: 'white',
          borderWidth: 2
        }]
      },

    });
