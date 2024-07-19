document.addEventListener('DOMContentLoaded', function () {
    var sliders = document.querySelectorAll('input[type=range]');
    sliders.forEach(function (slider) {
        var input = document.getElementById(slider.id.replace('-slider', ''));
        slider.addEventListener('input', function () {
            input.value = this.value;
        });
    });

    var inputs = document.querySelectorAll('input[type=text]');
    inputs.forEach(function (input) {
        var slider = document.getElementById(input.id + '-slider');
        input.addEventListener('input', function () {
            slider.value = this.value;
        });
    });
});
