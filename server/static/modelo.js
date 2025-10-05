// Variable global para guardar el nombre del modelo que el usuario seleccione
let modeloSeleccionado = null;

// Esperamos a que todo el contenido de la página se cargue
document.addEventListener('DOMContentLoaded', () => {
  
  // Seleccionamos TODOS los botones de "Seleccionar"
  const botonesSeleccionar = document.querySelectorAll('.model-card .btn-primary');
  // Seleccionamos el botón principal "Nuevo Proyecto"
  const botonNuevoProyecto = document.querySelector('.nuevo-proyecto');

  // Recorremos cada botón de las tarjetas
  botonesSeleccionar.forEach(boton => {
    boton.addEventListener('click', (evento) => {
      
      const tarjeta = evento.target.closest('.model-card');
      const nombreDelModelo = tarjeta.querySelector('.first-content span').textContent;
      
      // Guardamos el nombre en nuestra variable global
      modeloSeleccionado = nombreDelModelo;
      
      // Actualizamos el texto del botón principal para dar feedback al usuario
      botonNuevoProyecto.querySelector('span').textContent = `Crear Proyecto con: ${nombreDelModelo}`;
      botonNuevoProyecto.querySelector('i').classList.add('fa-plus'); // Aseguramos que tenga el ícono
      
      //alert('Modelo seleccionado: ' + nombreDelModelo);
    });
  });

  // Agregamos el evento de clic al botón principal
  botonNuevoProyecto.addEventListener('click', () => {
    // Si no se ha seleccionado un modelo, mostramos una alerta
    if (!modeloSeleccionado) {
      //lert('Por favor, selecciona un modelo de la lista.');
      return;
    }
    
    // Si hay un modelo, redirigimos al dashboard, pasando el nombre en la URL
    // JavaScript se encargará de codificar el nombre para que sea seguro en la URL
    window.location.href = `/dashboard/${modeloSeleccionado}`;
  });

});

// --- OTRAS FUNCIONES (puedes mantenerlas si las usas) ---
function filterFiles(){
  const q = document.getElementById('q').value.toLowerCase();
  const rows = document.querySelectorAll('#files tr');
  rows.forEach(r=>{
    const text = r.innerText.toLowerCase();
    r.style.display = text.includes(q) ? '' : 'none';
  });
}

// Script para el botón de comunidad
const botonComunidad = document.getElementById('botonComunidad');
botonComunidad.addEventListener('click',(el)=>{
  el.preventDefault();
  window.location.href="/community";
});