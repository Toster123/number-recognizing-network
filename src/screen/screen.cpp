#include "screen.hpp"
#include <QPainter>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QMouseEvent>
#include <QApplication>
#include <QMessageBox>
#include <iostream>


DrawingCanvas::DrawingCanvas(QWidget* parent) 
    : QWidget(parent), drawing_(false), pen_width_(40) {
    setFixedSize(500, 500);
    canvas_ = QPixmap(500, 500);
    canvas_.fill(Qt::white);
    setMouseTracking(false);
}

QImage DrawingCanvas::GetImage() const {
    return canvas_.toImage();
}

void DrawingCanvas::ClearCanvas() {
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

Screen::Screen(std::unique_ptr<SequentialNetwork> network, QWidget* parent)
    : QWidget(parent), network_(std::move(network)) {
    setWindowTitle("Number Recognition");
    setFixedSize(800, 540);
    SetupUI();
}

void Screen::SetupUI() {
    auto* main_layout = new QHBoxLayout(this);
    
    canvas_ = new DrawingCanvas(this);
    
    auto* central_layout = new QVBoxLayout();
    central_layout->addWidget(canvas_);
    central_layout->addStretch();

    main_layout->addLayout(central_layout);
    
    auto* right_layout = new QVBoxLayout();
    
    auto* results_grid = new QGridLayout();
    result_labels_.reserve(10);
    
    for (size_t i = 0; i < 10; ++i) {
        result_labels_[i] = new QLabel("...", this);
        result_labels_[i]->setMinimumWidth(150);
        result_labels_[i]->setStyleSheet("QLabel {padding: 5px;}");
        results_grid->addWidget(result_labels_[i], i, 0);
    }
    
    right_layout->addLayout(results_grid);
    
    recognize_button_ = new QPushButton("Recognize", this);
    clear_button_ = new QPushButton("Clear", this);
    
    connect(recognize_button_, &QPushButton::clicked, this, &Screen::PredictNumber);
    connect(clear_button_, &QPushButton::clicked, this, &Screen::ClearCanvas);
    
    auto* button_layout = new QHBoxLayout();
    button_layout->addWidget(clear_button_);
    button_layout->addWidget(recognize_button_);
    right_layout->addLayout(button_layout);
    
    status_label_ = new QLabel("", this);
    status_label_->setStyleSheet("QLabel {color: red;}");
    right_layout->addWidget(status_label_);
    
    right_layout->addStretch();
    main_layout->addLayout(right_layout);
    
    setLayout(main_layout);
}

void Screen::PredictNumber() {
    status_label_->setText("");
    
    try {
        QImage image = canvas_->GetImage();
        Matrix3D input = PreprocessImage(image);
        
        std::vector<double> result = network_->Feedforward(input);
        
        int predicted_number = std::distance(result.begin(), std::max_element(result.begin(), result.end()));
        
        for (int i = 0; i < 10; ++i) {
            int confidence = static_cast<int>(result[i] * 100);
            QString text = QString("%1, %2%").arg(i).arg(confidence);
            if (i == predicted_number) {
                text += " - ✅";
                result_labels_[i]->setStyleSheet("QLabel {padding: 5px; color: green; font-weight: bold;}");
            } else {
                result_labels_[i]->setStyleSheet("QLabel {padding: 5px; color: black;}");
            }
            result_labels_[i]->setText(text);
        }
        
        std::cout << "Result " << predicted_number << ", " 
                  << (result[predicted_number] * 100) << "%" << std::endl;
        
    } catch (const std::exception& e) {
        status_label_->setText("Error during recognition");
        std::cerr << e.what() << std::endl;
    }
}

void Screen::ClearCanvas() {
    canvas_->ClearCanvas();
    
    for (size_t i = 0; i < 10; ++i) {
        result_labels_[i]->setText("...");
        result_labels_[i]->setStyleSheet("QLabel {padding: 5px; color: black;}");
    }
    
    status_label_->setText("");
}

Matrix3D Screen::PreprocessImage(const QImage& image) {
    QImage scaled_image = image.scaled(28, 28, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    QImage gray_image = scaled_image.convertToFormat(QImage::Format_Grayscale8);
    
    Matrix3D result(1, Matrix2D(28, std::vector<double>(28)));
    
    for (int h = 0; h < 28; ++h) {
        for (int w = 0; w < 28; ++w) {
            QRgb pixel = gray_image.pixel(w, h);
            double value = qGray(pixel);
            
            value = 255.0 - value;
            
            value /= 255.0;
            
            result[0][h][w] = value;
        }
    }
    
    std::cout << "Image:" << std::endl;
    for (int h = 0; h < 28; ++h) {
        for (int w = 0; w < 28; ++w) {
            std::cout << result[0][h][w] << " ";
        }
        std::cout << std::endl;
    }
    
    return result;
}