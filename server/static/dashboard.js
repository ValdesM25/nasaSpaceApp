// dashboard.js
document.addEventListener('DOMContentLoaded', () => {
    // Seleccionamos los elementos del formulario por su ID
    const predictForm = document.getElementById('predict-form');
    const csvButton = document.getElementById('csv-button');
    const csvInput = document.getElementById('csv-input');
    const fileNameDisplay = document.getElementById('file-name-display');

    // Verificamos que todos los elementos existan
    if (predictForm && csvButton && csvInput && fileNameDisplay) {

        // Cuando el usuario haga clic en el botón visible "Subir CSV"...
        csvButton.addEventListener('click', () => {
            // ...hacemos clic en el input de archivo oculto.
            csvInput.click();
        });

        // Cuando el usuario seleccione un archivo...
        csvInput.addEventListener('change', () => {
            if (csvInput.files.length > 0) {
                // Mostramos el nombre del archivo
                fileNameDisplay.textContent = csvInput.files[0].name;

                // ¡Enviamos el formulario automáticamente!
                predictForm.submit();
            } else {
                fileNameDisplay.textContent = '';
            }
        });
    }
});