document.addEventListener('DOMContentLoaded', function () {
    var elems = document.querySelectorAll('select');
    var instances = M.FormSelect.init(elems);
});

document.getElementById('plotType').addEventListener('change', function (event) {
    event.preventDefault(); // Prevent default behavior
    var plotType = this.value;
    var var2Container = document.getElementById('var2-container');
    var fixed4Container = document.getElementById('fixed4-container');
    var fixed5Container = document.getElementById('fixed5-container');
    if (plotType === '3d') {
        var2Container.style.display = 'block';
        fixed4Container.style.display = 'block';
        fixed5Container.style.display = 'none';
    } else {
        var2Container.style.display = 'none';
        fixed4Container.style.display = 'block';
        fixed5Container.style.display = 'block';
    }
});

document.getElementById('outputVariable').addEventListener('change', function (event) {
    event.preventDefault(); // Prevent default behavior
});

document.getElementById('var1').addEventListener('change', function (event) {
    event.preventDefault(); // Prevent default behavior
    updateFixedVariables();
});

document.getElementById('var2').addEventListener('change', function (event) {
    event.preventDefault(); // Prevent default behavior
    updateFixedVariables();
});

function updateFixedVariables() {
    var var1 = document.getElementById('var1').value;
    var var2 = document.getElementById('var2').value;
    var allVars = ['HCC', 'WCC', 'LCC', 'Tamb', 'Uin', 'Q'];

    allVars.forEach(function (v) {
        document.getElementById('fixed-' + v).style.display = 'none';
    });

    allVars.forEach(function (v) {
        if (v !== var1 && v !== var2) {
            document.getElementById('fixed-' + v).style.display = 'block';
        }
    });
}

function generatePlot() {
    var plotType = document.getElementById('plotType').value;
    var outputVariable = document.getElementById('outputVariable').value;
    var var1 = document.getElementById('var1').value;
    var var2 = document.getElementById('var2').value;
    var fixedValues = {};

    ['HCC', 'WCC', 'LCC', 'Tamb', 'Uin', 'Q'].forEach(function (v) {
        var fixedInput = document.getElementById('fixed-' + v).querySelector('input[type="text"]').value;
        fixedValues[v] = fixedInput !== '' ? parseFloat(fixedInput) : null;
    });

    // Ensure all fields are filled out except the independent variables
    if (!plotType || !outputVariable || !var1 || (plotType === '3d' && !var2) || Object.keys(fixedValues).some(k => (fixedValues[k] === null && k !== var1 && k !== var2))) {
        alert('Please fill all fields for the selected plot type.');
        return;
    }

    var plotData = {
        plotType: plotType,
        outputVariable: outputVariable,
        var1: var1,
        var2: var2,
        fixedValues: fixedValues
    };

    console.log("Plot Data:", plotData);

    fetch('/plot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(plotData),
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                var plotDiv = document.getElementById('plot');
                plotDiv.innerHTML = '';
                var plotData = JSON.parse(data.plotData);
                Plotly.newPlot('plot', plotData.data, plotData.layout);
            } else {
                alert('Error generating plot: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error generating plot: ' + error);
        });
}

// Update slider and input value together
document.querySelectorAll('.slider-input-container input[type="range"]').forEach(slider => {
    slider.addEventListener('input', function (event) {
        event.preventDefault(); // Prevent default behavior
        const input = this.nextElementSibling;
        input.value = this.value;
    });
});

document.querySelectorAll('.slider-input-container input[type="text"]').forEach(input => {
    input.addEventListener('input', function (event) {
        event.preventDefault(); // Prevent default behavior
        const slider = this.previousElementSibling;
        if (parseFloat(this.value) >= parseFloat(slider.min) && parseFloat(this.value) <= parseFloat(slider.max)) {
            slider.value = this.value;
        }
    });
});
