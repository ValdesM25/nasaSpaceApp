let modeloSeleccionado = null;

function seleccionarModelo(nombre) {
  modeloSeleccionado = nombre;
  alert('Modelo seleccionado: ' + nombre);
}

function nuevoProyecto() {
  if (!modeloSeleccionado) {
    alert('Debes seleccionar un modelo antes de crear un nuevo proyecto.');
    return;
  }
  alert('Nuevo proyecto creado usando el modelo: ' + modeloSeleccionado);
}

function toggleView(){ alert('Abrir panel de filtros (simulación)'); }
function upload(){ alert('Seleccionar archivo para subir (simulación)'); }
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

