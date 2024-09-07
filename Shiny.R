dlibrary(shiny)
library(randomForest)
f <- read.csv("Sleep_Efficiency_Updated.csv")
# Load the random forest model from the .rds file
rf_model <- readRDS("rf_model.rds")
f$Smoking.status <- as.factor(f$Smoking.status)
# Define UI
ui <- fluidPage(
  titlePanel("Sleep Efficiency Calculator"),
  sidebarLayout(
    sidebarPanel(
      numericInput("light_sleep", "Light Sleep Percentage:", value = 20, min = 0, max = 100),
      numericInput("deep_sleep", "Deep Sleep Percentage:", value = 30, min = 0, max = 100),
      numericInput("awakenings", "Number of Awakenings:", value = 2, min = 0),
      numericInput("alcohol_consumption", "Alcohol Consumption:", value = 2, min = 0),
      numericInput("age", "Age:", value = 30, min = 0),
      selectInput("smoking_status", "Smoking Status:", choices = c("Yes", "No")),
      numericInput("exercise_frequency", "Exercise Frequency:", value = 3, min = 0),
      numericInput("rem_sleep", "REM Sleep Percentage:", value = 20, min = 0, max = 100),
      actionButton("calculate_button", "Calculate Sleep Efficiency")
    ),
    mainPanel(
      h3("Results"),
      verbatimTextOutput("sleep_efficiency_result")
    )
  )
)

server <- function(input, output) {
  observeEvent(input$calculate_button, {
    # Convert selectInput choice to 1 or 0
    
    # Prepare input data
    input_data <- data.frame(
      Light.sleep.percentage = input$light_sleep,
      Deep.sleep.percentage = input$deep_sleep,
      Awakenings = input$awakenings,
      Alcohol.consumption = input$alcohol_consumption,
      Age = input$age,
      Smoking.status = factor(input$smoking_status, levels = c("Yes", "No")),
      Exercise.frequency = input$exercise_frequency,
      REM.sleep.percentage = input$rem_sleep
    )
    
    # Print input data for debugging
    print(input_data)
    
    # Predict sleep efficiency using the random forest model
    sleep_efficiency <- predict(rf_model, newdata = input_data)
    
    # Output sleep efficiency result
    output$sleep_efficiency_result <- renderText({
      paste("Predicted Sleep Efficiency:", sleep_efficiency)
    })
  })
}


# Run the application
shinyApp(ui = ui, server = server)