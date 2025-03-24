// Document ready function
document.addEventListener("DOMContentLoaded", function () {
  // Initialize form validation
  initFormValidation();

  // Initialize theme toggle
  initThemeToggle();

  // Initialize form handlers
  initFormHandlers();
});

/**
 * Initializes form validation for all forms with class "needs-validation"
 * Prevents form submission if validation fails and shows loading spinner when valid
 */
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

/**
 * Sets up the light/dark theme toggle functionality
 * Loads saved preference from localStorage and applies it
 */
function initThemeToggle() {
  const themeToggleBtn = document.getElementById("theme-toggle");
  const themeIcon = themeToggleBtn.querySelector("i");

  // Check for saved theme preference
  const savedTheme = localStorage.getItem("theme");
  if (savedTheme) {
    document.body.setAttribute("data-bs-theme", savedTheme);
    updateThemeIcon(themeIcon, savedTheme);
  }

  // Add event listener to toggle theme
  themeToggleBtn.addEventListener("click", () => {
    const currentTheme = document.body.getAttribute("data-bs-theme");
    const newTheme = currentTheme === "dark" ? "light" : "dark";

    // Save theme preference
    localStorage.setItem("theme", newTheme);
    document.body.setAttribute("data-bs-theme", newTheme);
    updateThemeIcon(themeIcon, newTheme);

    // Also update plot theme if plot exists
    if (window.currentPlot) {
      updatePlotTheme(newTheme);
    }
  });
}

/**
 * Updates the theme toggle icon based on current theme
 * @param {HTMLElement} iconElement - The icon element to update
 * @param {string} theme - The theme value ('light' or 'dark')
 */
function updateThemeIcon(iconElement, theme) {
  if (theme === "dark") {
    iconElement.classList.remove("bi-moon-fill");
    iconElement.classList.add("bi-sun-fill");
  } else {
    iconElement.classList.remove("bi-sun-fill");
    iconElement.classList.add("bi-moon-fill");
  }
}

/**
 * Initializes all form-related event handlers for the visualization section
 */
function initFormHandlers() {
  // Connect visualization form elements
  const plotForm = document.getElementById("visualization-form");
  if (plotForm) {
    const plotTypeRadios = document.querySelectorAll('input[name="plotType"]');
    const var1Select = document.getElementById("var1");
    const var2Select = document.getElementById("var2");
    const var2Group = document.getElementById("var2Group");

    // Handle plot type change
    plotTypeRadios.forEach((radio) => {
      radio.addEventListener("change", () => {
        if (radio.value === "3d") {
          var2Group.classList.remove("d-none");
        } else {
          var2Group.classList.add("d-none");
        }
      });
    });

    // Handle variable 1 change to prevent duplicate selections
    var1Select.addEventListener("change", () => {
      updateFixedVariables();
    });

    // Handle variable 2 change to prevent duplicate selections
    var2Select.addEventListener("change", () => {
      updateFixedVariables();
    });

    // Submit event for visualization form
    plotForm.addEventListener("submit", (e) => {
      e.preventDefault();
      generatePlot();
    });
  }

  // Connect sliders with input fields
  connectSlidersWithInputs();
}

/**
 * Connects range sliders with their corresponding text input fields
 * Sets up two-way binding between slider and input values
 */
function connectSlidersWithInputs() {
  const sliders = document.querySelectorAll('input[type="range"]');

  sliders.forEach((slider) => {
    const inputId = slider.getAttribute("data-input");
    const input = document.getElementById(inputId);

    if (input) {
      // Set initial value from input to slider
      slider.value = input.value;

      // Update input when slider changes
      slider.addEventListener("input", () => {
        input.value = slider.value;
      });

      // Update slider when input changes
      input.addEventListener("input", () => {
        slider.value = input.value;
      });
    }
  });
}

/**
 * Updates the fixed variables section based on variable selections
 * Hides the fixed variable inputs for variables selected as var1 or var2
 */
function updateFixedVariables() {
  const var1 = document.getElementById("var1").value;
  const var2 = document.getElementById("var2").value;
  const plotType = document.querySelector(
    'input[name="plotType"]:checked'
  ).value;

  // Show all fixed variable groups first
  const fixedGroups = document.querySelectorAll(".fixed-var-group");
  fixedGroups.forEach((group) => {
    group.classList.remove("d-none");
  });

  // Hide the selected variables
  if (var1) {
    const var1Group = document.getElementById(`fixed-${var1}`);
    if (var1Group) {
      var1Group.classList.add("d-none");
    }
  }

  if (plotType === "3d" && var2) {
    const var2Group = document.getElementById(`fixed-${var2}`);
    if (var2Group) {
      var2Group.classList.add("d-none");
    }
  }
}

/**
 * Generates and displays a plot based on form inputs
 * Sends request to server and renders the returned plot data
 */
function generatePlot() {
  // Get form values
  const plotType = document.querySelector(
    'input[name="plotType"]:checked'
  ).value;
  const outputVariable = document.querySelector(
    'input[name="outputVariable"]:checked'
  ).value;
  const var1 = document.getElementById("var1").value;
  const var2 = document.getElementById("var2").value;

  // Validate required fields
  if (!var1) {
    showValidationError("Please select Variable 1");
    return;
  }

  if (plotType === "3d" && !var2) {
    showValidationError("Please select Variable 2 for 3D plots");
    return;
  }

  if (var1 === var2 && plotType === "3d") {
    showValidationError("Variables 1 and 2 must be different for 3D plots");
    return;
  }

  // Collect fixed values
  const fixedValues = {};
  const fixedInputs = document.querySelectorAll(
    ".fixed-var-input:not(.d-none)"
  );
  fixedInputs.forEach((input) => {
    const varName = input.getAttribute("data-var");
    fixedValues[varName] = input.value;
  });

  // Show loading indicator
  const plotLoading = document.getElementById("plot-loading");
  const plotContainer = document.getElementById("plot-container");
  const plotError = document.getElementById("plot-error");

  plotError.classList.add("d-none");
  plotLoading.classList.remove("d-none");
  plotContainer.classList.add("d-none");

  // Prepare data for the request
  const requestData = {
    plotType: plotType,
    outputVariable: outputVariable,
    var1: var1,
    fixedValues: fixedValues,
  };

  if (plotType === "3d") {
    requestData.var2 = var2;
  }

  // Send request to the server
  fetch("/plot", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestData),
  })
    .then((response) => response.json())
    .then((data) => {
      // Hide loading indicator
      plotLoading.classList.add("d-none");

      if (data.success) {
        // Parse the plot data
        const plotData = JSON.parse(data.plotData);

        // Apply theme to the plot
        const theme = document.body.getAttribute("data-bs-theme") || "light";
        applyPlotTheme(plotData, theme);

        // Create the plot
        Plotly.newPlot("plot-div", plotData.data, plotData.layout, {
          responsive: true,
        });

        // Store the current plot data for theme updates
        window.currentPlot = plotData;

        // Show the plot container
        plotContainer.classList.remove("d-none");
      } else {
        // Show error
        plotError.textContent = data.error || "Error generating plot";
        plotError.classList.remove("d-none");
      }
    })
    .catch((error) => {
      // Hide loading indicator and show error
      plotLoading.classList.add("d-none");
      plotError.textContent = "Error: " + error.message;
      plotError.classList.remove("d-none");
    });
}

/**
 * Applies theme-specific styling to plot configuration
 * @param {Object} plotConfig - The plot configuration object
 * @param {string} theme - The theme to apply ('light' or 'dark')
 */
function applyPlotTheme(plotConfig, theme) {
  // Default colors
  const textColor = theme === "dark" ? "#fff" : "#333";
  const gridColor =
    theme === "dark" ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)";
  const paperBgColor = theme === "dark" ? "#212529" : "#fff";
  const plotBgColor = theme === "dark" ? "#212529" : "#fff";

  // Apply to layout
  if (plotConfig.layout) {
    // Set font color
    plotConfig.layout.font = {
      color: textColor,
    };

    // Set background colors
    plotConfig.layout.paper_bgcolor = paperBgColor;
    plotConfig.layout.plot_bgcolor = plotBgColor;

    // Style 2D axes
    if (plotConfig.layout.xaxis) {
      plotConfig.layout.xaxis.gridcolor = gridColor;
      plotConfig.layout.xaxis.linecolor = textColor;
    }

    if (plotConfig.layout.yaxis) {
      plotConfig.layout.yaxis.gridcolor = gridColor;
      plotConfig.layout.yaxis.linecolor = textColor;
    }

    // Style 3D axes
    if (plotConfig.layout.scene) {
      ["xaxis", "yaxis", "zaxis"].forEach((axis) => {
        if (plotConfig.layout.scene[axis]) {
          plotConfig.layout.scene[axis].gridcolor = gridColor;
          plotConfig.layout.scene[axis].linecolor = textColor;
        }
      });
    }
  }
}

/**
 * Updates the theme of the current plot
 * @param {string} theme - The theme to apply ('light' or 'dark')
 */
function updatePlotTheme(theme) {
  if (!window.currentPlot) return;

  // Apply theme to the stored plot data
  applyPlotTheme(window.currentPlot, theme);

  // Update the plot
  Plotly.react("plot-div", window.currentPlot.data, window.currentPlot.layout, {
    responsive: true,
  });
}

/**
 * Shows a validation error message
 * @param {string} message - The error message to display
 */
function showValidationError(message) {
  const errorAlert = document.getElementById("plot-error");
  errorAlert.textContent = message;
  errorAlert.classList.remove("d-none");
}
