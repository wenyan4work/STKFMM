#include <iostream>
#include <string>

namespace Laplace1D3D {
int main(int argc, char *argv[]);
}

namespace Laplace2D3D {
int main(int argc, char *argv[]);
}

namespace Laplace3D3D {
int main(int argc, char *argv[]);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Input: {dim} {N}.\n";
        return 1;
    }

    int dim = atoi(argv[1]);
    argc--;
    argv++;

    switch (dim) {
    case 1:
        Laplace1D3D::main(argc, argv);
        break;
    case 2:
        Laplace2D3D::main(argc, argv);
        break;
    case 3:
        Laplace3D3D::main(argc, argv);
        break;
    default:
        std::cerr << "Dimension {" << dim << "} not supported for charge.\n";
        return 1;
    }
    return 0;
}
