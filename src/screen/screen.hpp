#pragma once

#include "../neural_network/network.hpp"
#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QPainter>
#include <QMouseEvent>
#include <QPixmap>
#include <QImage>
#include <memory>
#include <vector>

class DrawingCanvas : public QWidget {
    Q_OBJECT

public:
    explicit DrawingCanvas(QWidget* parent = nullptr);
    QImage getImage() const;
    void clearCanvas();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    bool drawing_;
    QPixmap canvas_;
    QPoint last_point_;
    int pen_width_;
};

class Screen : public QWidget {
    Q_OBJECT

public:
    explicit Screen(std::unique_ptr<SequentialNetwork> network, QWidget* parent = nullptr);

private slots:
    void predict_number();
    void clear_canvas();

private:
    void setup_ui();
    Matrix3D preprocess_image(const QImage& image);
    
    std::unique_ptr<SequentialNetwork> network_;
    DrawingCanvas* canvas_;
    std::vector<QLabel*> result_labels_;
    QPushButton* recognize_button_;
    QPushButton* clear_button_;
    QLabel* status_label_;
};