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
// class Environment;
// class Muscle;
class exo_Window : public Window 
{
public:
    exo_Window(Environment*, const std::string&, const std::string&, const std::string&);
    void Step()override;
    Eigen::VectorXd GetExoTorquesFromNN();
    void record_data()override;
    void define_muscle_groups()override;
    void Plot_And_Save()override;

    py::object exo_agent;
    py::object plotter;

    // enum MuscleGroupIndex {
    //     LHFlex,
    //     LHExt,
    //     LHAbd,
    //     LHAdd,
    //     LHExtRot,
    //     LHIntRot,
    //     LKFlex,
    //     LKExt,
    //     RHFlex,
    //     RHExt,
    //     RHAbd,
    //     RHAdd,
    //     RHExtRot,
    //     RHIntRot,
    //     RKFlex,
    //     RKExt,
    //     TOTAL
    // };

    // struct MuscleGroup {
    //     double total_activation;
    //     std::unordered_set<std::string> group;

    //     inline double add(double activation) {
    //         total_activation += activation;
    //         return total_activation;
    //     }

    //     inline double get_avg_activation() {
    //         return total_activation / group.size();
    //     }
    // };

    // std::array<MuscleGroup, MuscleGroupIndex::TOTAL> muscle_groups;
};
}