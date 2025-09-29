#include "screen.hpp"
#include <QPainter>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QMouseEvent>
#include <QApplication>
#include <QMessageBox>
#include <iostream>
// #include <algorithm>

// DrawingCanvas implementation
DrawingCanvas::DrawingCanvas(QWidget* parent) 
    : QWidget(parent), drawing_(false), pen_width_(40) {
    setFixedSize(500, 500);
    canvas_ = QPixmap(500, 500);
    canvas_.fill(Qt::white);
    setMouseTracking(false);
}

QImage DrawingCanvas::getImage() const {
    return canvas_.toImage();
}

void DrawingCanvas::clearCanvas() {
    canvas_.fill(Qt::white);
    update();
}

void DrawingCanvas::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event)
    QPainter painter(this);
    painter.drawPixmap(0, 0, canvas_);
}

void DrawingCanvas::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        drawing_ = true;
        last_point_ = event->pos();
    }
}

void DrawingCanvas::mouseMoveEvent(QMouseEvent* event) {
    if (event->buttons() & Qt::LeftButton && drawing_) {
        QPainter painter(&canvas_);
        painter.setPen(QPen(Qt::black, pen_width_, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        painter.drawLine(last_point_, event->pos());
        last_point_ = event->pos();
        update();
    }
}

void DrawingCanvas::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && drawing_) {
        drawing_ = false;
    }
}

// Screen implementation
Screen::Screen(std::unique_ptr<SequentialNetwork> network, QWidget* parent)
    : QWidget(parent), network_(std::move(network)) {
    setWindowTitle("Number Recognition");
    setFixedSize(800, 540);
    setup_ui();
}

void Screen::setup_ui() {
    auto* main_layout = new QHBoxLayout(this);
    
    canvas_ = new DrawingCanvas(this);
    
    auto* central_panel = new QVBoxLayout();
    central_panel->addWidget(canvas_);
    central_panel->addStretch();

    main_layout->addLayout(central_panel);
    
    // Right panel with results and controls
    auto* right_panel = new QVBoxLayout();
    
    // Results grid
    auto* results_grid = new QGridLayout();
    result_labels_.resize(10);
    
    for (int i = 0; i < 10; ++i) {
        result_labels_[i] = new QLabel("...", this);
        result_labels_[i]->setMinimumWidth(150);
        result_labels_[i]->setStyleSheet("QLabel { padding: 5px; }");
        results_grid->addWidget(result_labels_[i], i, 0);
    }
    
    right_panel->addLayout(results_grid);
    
    // Buttons
    recognize_button_ = new QPushButton("Recognize", this);
    clear_button_ = new QPushButton("Clear", this);
    
    connect(recognize_button_, &QPushButton::clicked, this, &Screen::predict_number);
    connect(clear_button_, &QPushButton::clicked, this, &Screen::clear_canvas);
    
    auto* button_layout = new QHBoxLayout();
    button_layout->addWidget(clear_button_);
    button_layout->addWidget(recognize_button_);
    right_panel->addLayout(button_layout);
    
    // Status label
    status_label_ = new QLabel("", this);
    status_label_->setStyleSheet("QLabel { color: red; }");
    right_panel->addWidget(status_label_);
    
    // Add stretch to push everything up
    right_panel->addStretch();
    
    main_layout->addLayout(right_panel);
    
    setLayout(main_layout);
}

void Screen::predict_number() {
    status_label_->setText("");
    
    try {
        // Get image from canvas
        QImage image = canvas_->getImage();
        
        // Preprocess the image
        Matrix3D processed_input = preprocess_image(image);
        
        // Get prediction from network
        std::vector<double> result = network_->Feedforward(processed_input);
        
        // Find predicted number
        int predicted_number = std::distance(result.begin(), 
                                           std::max_element(result.begin(), result.end()));
        
        // Update result labels
        for (int i = 0; i < 10; ++i) {
            int percentage = static_cast<int>(result[i] * 100);
            QString text = QString("%1, %2%").arg(i).arg(percentage);
            if (i == predicted_number) {
                text += " - ✅";
                result_labels_[i]->setStyleSheet("QLabel { padding: 5px; color: green; font-weight: bold; }");
            } else {
                result_labels_[i]->setStyleSheet("QLabel { padding: 5px; color: black; }");
            }
            result_labels_[i]->setText(text);
        }
        
        std::cout << "Result " << predicted_number << ", " 
                  << (result[predicted_number] * 100) << "%" << std::endl;
        
    } catch (const std::exception& e) {
        status_label_->setText("Error");
        std::cerr << e.what() << std::endl;
    }
}

void Screen::clear_canvas() {
    canvas_->clearCanvas();
    
    // Reset result labels
    for (int i = 0; i < 10; ++i) {
        result_labels_[i]->setText("...");
        result_labels_[i]->setStyleSheet("QLabel { padding: 5px; color: black; }");
    }
    
    status_label_->setText("");
}

Matrix3D Screen::preprocess_image(const QImage& image) {
    // Convert to 28x28 grayscale
    QImage scaled_image = image.scaled(28, 28, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    QImage gray_image = scaled_image.convertToFormat(QImage::Format_Grayscale8);
    
    // Create 3D matrix (1 channel, 28x28)
    Matrix3D result(1, Matrix2D(28, std::vector<double>(28)));
    
    for (int h = 0; h < 28; ++h) {
        for (int w = 0; w < 28; ++w) {
            // Get pixel value (0-255)
            QRgb pixel = gray_image.pixel(w, h);
            double gray_value = qGray(pixel);
            
            // Invert colors (white background becomes black, black drawing becomes white)
            gray_value = 255.0 - gray_value;
            
            // Normalize to [0, 1]
            gray_value /= 255.0;
            
            result[0][h][w] = gray_value;
        }
    }
    
    // Print all elements of the result matrix
    std::cout << "Matrix elements:" << std::endl;
    for (int h = 0; h < 28; ++h) {
        for (int w = 0; w < 28; ++w) {
            std::cout << result[0][h][w] << " ";
        }
        std::cout << std::endl;
    }
    
    return result;
}

