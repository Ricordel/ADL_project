#include <sstream>
#include "Function.hpp"

Function::~Function() {}



Function_a_b_cd::Function_a_b_cd()
        : m_a(0), m_b(0), m_c(0), m_d(0)
{}

Function_a_b_cd::Function_a_b_cd(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
        : m_a(a), m_b(b), m_c(c), m_d(d)
{}


Function_a_b_cd::~Function_a_b_cd() {}


std::string Function_a_b_cd::toString() const
{
        std::ostringstream sstr;
        sstr << m_a  << "," << m_b << ",(" << m_c << "," << m_d << ")" << std::endl;

        return sstr.str();
}


std::string Function_a_b_cd::toPrettyString() const
{
        std::stringstream sstr;
        sstr << m_a  << " + " << m_b << " + " << m_c << "." << m_d << std::endl;

        return sstr.str();
}
