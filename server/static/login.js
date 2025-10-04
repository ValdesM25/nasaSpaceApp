const submit = document.getElementById('submit');
let usuarioEl = document.getElementById('username');
let passwordEl = document.getElementById('password');

submit.addEventListener('click', (el) => {
  el.preventDefault();

  const usuario = usuarioEl.value;
  const password = passwordEl.value;

  if (usuario == "admin" && password == "123") {
    //window.location.href = "../another-page.html"; // Redirects one level up
    window.location.href = "/inicio"; // Redirects relative to the domain root
  } else {
    console.log("usuario: admin, password: 123");
    window.location.href = "/inicio"; // Redirects relative to the domain root
  }
});
