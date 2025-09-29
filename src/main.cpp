#include "screen/screen.hpp"
#include <QApplication>
#include <memory>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    
    Screen window(std::make_unique<SequentialNetwork>());
    
    window.show();
    return app.exec();
}
