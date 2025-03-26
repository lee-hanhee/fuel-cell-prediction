// Document ready function
document.addEventListener("DOMContentLoaded", function () {
  // Initialize form validation
  initFormValidation();

  // Initialize theme toggle
  initThemeToggle();

  // Initialize form handlers
  initFormHandlers();
});

// Initialize form validation
function initFormValidation() {
  // Fetch all forms that we want to apply validation to
  const forms = document.querySelectorAll(".needs-validation");

  // Loop over them and prevent submission
  Array.from(forms).forEach((form) => {
    form.addEventListener(
      "submit",
      (event) => {
        if (!form.checkValidity()) {
          event.preventDefault();
          event.stopPropagation();
        } else {
          // Show loading spinner if form is valid
          const spinner = form.querySelector(".spinner-border");
          if (spinner) {
            spinner.classList.remove("d-none");
          }
        }
        form.classList.add("was-validated");
      },
      false
    );
  });
}

// Initialize theme toggle
function initThemeToggle() {
  const themeToggleBtn = document.getElementById("theme-toggle");
  const themeIcon = themeToggleBtn.querySelector("i");

  // Check for saved theme preference
  const savedTheme = localStorage.getItem("theme");
  if (savedTheme) {
    document.body.setAttribute("data-bs-theme", savedTheme);
    updateThemeIcon(themeIcon, savedTheme);
  }

  // Add event listener to theme toggle button
  themeToggleBtn.addEventListener("click", function () {
    const currentTheme = document.body.getAttribute("data-bs-theme");
    const newTheme = currentTheme === "dark" ? "light" : "dark";

    // Update theme
    document.body.setAttribute("data-bs-theme", newTheme);
    localStorage.setItem("theme", newTheme);

    // Update icon
    updateThemeIcon(themeIcon, newTheme);

    // Update Plotly theme if there's an existing plot
    const plotElement = document.getElementById("plot");
    if (plotElement && plotElement.data) {
      updatePlotTheme(newTheme);
    }

    // Also update charts if the updateChartsTheme function exists
    if (typeof updateChartsTheme === "function") {
      updateChartsTheme(newTheme === "dark");
    }
  });
}

// Update theme icon
function updateThemeIcon(iconElement, theme) {
  if (theme === "dark") {
    iconElement.classList.remove("bi-moon-fill");
    iconElement.classList.add("bi-sun-fill");
  } else {
    iconElement.classList.remove("bi-sun-fill");
    iconElement.classList.add("bi-moon-fill");
  }
}

// Initialize form event handlers
function initFormHandlers() {
  // Add event listeners to plot type select
  const plotTypeSelect = document.getElementById("plotType");
  if (plotTypeSelect) {
    plotTypeSelect.addEventListener("change", function () {
      const var2Container = document.getElementById("var2-container");
      if (this.value === "3d") {
        var2Container.classList.remove("d-none");
      } else {
        var2Container.classList.add("d-none");
      }
    });
  }

  // Add event listeners to var1 and var2 selects
  const var1Select = document.getElementById("var1");
  const var2Select = document.getElementById("var2");

  if (var1Select) {
    var1Select.addEventListener("change", updateFixedVariables);
  }

  if (var2Select) {
    var2Select.addEventListener("change", updateFixedVariables);
  }

  // Connect sliders with input fields
  connectSlidersWithInputs();
}

// Connect range sliders with text inputs
function connectSlidersWithInputs() {
  // Main form sliders
  const sliderIds = ["HCC", "WCC", "LCC", "Tamb", "Uin", "Q"];

  sliderIds.forEach((id) => {
    const slider = document.getElementById(`${id}-slider`);
    const input = document.getElementById(id);

    if (slider && input) {
      // Set initial values
      input.value = slider.value;

      // Update input when slider changes
      slider.addEventListener("input", function () {
        input.value = this.value;
      });

      // Update slider when input changes
      input.addEventListener("input", function () {
        const value = parseFloat(this.value);
        const min = parseFloat(slider.min);
        const max = parseFloat(slider.max);

        if (!isNaN(value) && value >= min && value <= max) {
          slider.value = value;
        }
      });
    }
  });
}

// Update fixed variables based on selected variables
function updateFixedVariables() {
  const var1 = document.getElementById("var1").value;
  const var2 = document.getElementById("var2").value;
  const allVars = ["HCC", "WCC", "LCC", "Tamb", "Uin", "Q"];

  // Hide all fixed variables first
  allVars.forEach((varName) => {
    const fixedVarElement = document.getElementById(`fixed-${varName}`);
    if (fixedVarElement) {
      fixedVarElement.classList.add("d-none");
    }
  });

  // Show only the relevant fixed variables
  allVars.forEach((varName) => {
    if (varName !== var1 && varName !== var2) {
      const fixedVarElement = document.getElementById(`fixed-${varName}`);
      if (fixedVarElement) {
        fixedVarElement.classList.remove("d-none");
      }
    }
  });
}

// Generate plot based on form input
function generatePlot() {
  // Get form values
  const plotType = document.getElementById("plotType").value;
  const outputVariable = document.getElementById("outputVariable").value;
  const var1 = document.getElementById("var1").value;
  const var2 = document.getElementById("var2").value;

  // Validate required fields
  if (!plotType || !outputVariable || !var1 || (plotType === "3d" && !var2)) {
    showValidationError("Please select all required fields");
    return;
  }

  // Get fixed values
  const fixedValues = {};
  const allVars = ["HCC", "WCC", "LCC", "Tamb", "Uin", "Q"];

  allVars.forEach((varName) => {
    const fixedInput = document.getElementById(`fixed${varName}`);
    if (fixedInput && varName !== var1 && varName !== var2) {
      const value = fixedInput.value.trim();

      if (!value) {
        showValidationError(`Please enter a value for fixed ${varName}`);
        return;
      }

      fixedValues[varName] = parseFloat(value);
    }
  });

  // Show loading spinner
  const spinner = document.getElementById("plot-loading-spinner");
  if (spinner) {
    spinner.classList.remove("d-none");
  }

  // Disable generate button during request
  const generateBtn = document.getElementById("generate-plot-btn");
  if (generateBtn) {
    generateBtn.disabled = true;
  }

  // Prepare data for API request
  const plotData = {
    plotType: plotType,
    outputVariable: outputVariable,
    var1: var1,
    var2: plotType === "3d" ? var2 : null,
    fixedValues: fixedValues,
  };

  // Send API request
  fetch("/plot", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(plotData),
  })
    .then((response) => response.json())
    .then((data) => {
      // Hide loading spinner
      if (spinner) {
        spinner.classList.add("d-none");
      }

      // Enable generate button
      if (generateBtn) {
        generateBtn.disabled = false;
      }

      if (data.success) {
        const plotContainer = document.getElementById("plot");
        if (plotContainer) {
          // Clear previous plot
          plotContainer.innerHTML = "";

          // Parse plot data
          const plotConfig = JSON.parse(data.plotData);

          // Apply theme to plot
          applyPlotTheme(plotConfig);

          // Render plot
          Plotly.newPlot("plot", plotConfig.data, plotConfig.layout, {
            responsive: true,
          });

          // Scroll to the plot
          document.getElementById("plot-container").scrollIntoView({
            behavior: "smooth",
          });
        }
      } else {
        showValidationError("Error generating plot: " + data.error);
      }
    })
    .catch((error) => {
      // Hide loading spinner
      if (spinner) {
        spinner.classList.add("d-none");
      }

      // Enable generate button
      if (generateBtn) {
        generateBtn.disabled = false;
      }

      showValidationError("Error generating plot: " + error);
      console.error("Error:", error);
    });
}

// Apply theme to plot based on current site theme
function applyPlotTheme(plotConfig) {
  const currentTheme = document.body.getAttribute("data-bs-theme");

  if (currentTheme === "dark") {
    // Apply dark theme to plot
    if (plotConfig.layout) {
      plotConfig.layout.paper_bgcolor = "#2d2d2d";
      plotConfig.layout.plot_bgcolor = "#1e1e1e";
      plotConfig.layout.font = {
        color: "#ffffff",
      };

      // Update axis colors
      if (plotConfig.layout.xaxis) {
        plotConfig.layout.xaxis.gridcolor = "#444";
        plotConfig.layout.xaxis.linecolor = "#555";
      }

      if (plotConfig.layout.yaxis) {
        plotConfig.layout.yaxis.gridcolor = "#444";
        plotConfig.layout.yaxis.linecolor = "#555";
      }

      if (plotConfig.layout.scene) {
        if (plotConfig.layout.scene.xaxis) {
          plotConfig.layout.scene.xaxis.gridcolor = "#444";
          plotConfig.layout.scene.xaxis.linecolor = "#555";
        }

        if (plotConfig.layout.scene.yaxis) {
          plotConfig.layout.scene.yaxis.gridcolor = "#444";
          plotConfig.layout.scene.yaxis.linecolor = "#555";
        }

        if (plotConfig.layout.scene.zaxis) {
          plotConfig.layout.scene.zaxis.gridcolor = "#444";
          plotConfig.layout.scene.zaxis.linecolor = "#555";
        }
      }
    }
  }
}

// Update existing plot to match theme
function updatePlotTheme(theme) {
  const plotElement = document.getElementById("plot");
  if (plotElement && plotElement.data) {
    const update = {};

    if (theme === "dark") {
      update.layout = {
        paper_bgcolor: "#2d2d2d",
        plot_bgcolor: "#1e1e1e",
        font: { color: "#ffffff" },
        xaxis: { gridcolor: "#444", linecolor: "#555" },
        yaxis: { gridcolor: "#444", linecolor: "#555" },
      };
    } else {
      update.layout = {
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        font: { color: "#000000" },
        xaxis: { gridcolor: "#eee", linecolor: "#ccc" },
        yaxis: { gridcolor: "#eee", linecolor: "#ccc" },
      };
    }

    Plotly.update("plot", {}, update.layout);
  }
}

// Show validation error
function showValidationError(message) {
  // Use Bootstrap's toast or alert for showing errors
  // For simplicity, we'll use alert for now
  alert(message);
}
