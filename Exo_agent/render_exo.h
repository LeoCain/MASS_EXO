#include "/home/medicalrobotics/MASS_EXO/render/Window.h"
#include "dart/dart.hpp"
#include "dart/gui/gui.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
namespace py = pybind11;
namespace MASS
{
class exo_Window : public Window 
{
public:
    exo_Window(Environment*, const std::string&, const std::string&, const std::string&);
    void Step()override;
    Eigen::VectorXd GetExoTorquesFromNN();
    void record_data()override;
    void define_muscle_groups()override;
    void Plot_And_Save()override;

    py::object exo_agent;   //pybind object for torque actor
    py::object plotter;     // pybind object for plotter class
};
}