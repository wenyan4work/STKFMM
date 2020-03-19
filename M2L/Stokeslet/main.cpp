#include <iostream>

namespace Stokes1D3D {
int main(int argc, char *argv[]);
}

namespace Stokes2D3D {
int main(int argc, char *argv[]);
}

namespace Stokes3D3D {
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
        Stokes1D3D::main(argc, argv);
        break;
    case 2:
        Stokes2D3D::main(argc, argv);
        break;
    case 3:
        Stokes3D3D::main(argc, argv);
        break;
    default:
        std::cerr << "Invalid dimension {" << dim << "}.\n";
        return 1;
    }

    return 0;
}
