document.getElementById('sendRequestBtn').addEventListener('click', () => {
    fetch('/dfres', {
        method: 'GET',  // o 'GET', según tu endpoint
        headers: {
            'Content-Type': 'application/json'
        },
        
    })
    .then(response => response.text())
    .then(data => {
        //console.log('Response:', data);
        alert(data);  // opcional, puedes mostrar un mensaje bonito
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Something went wrong.');
    });
});
