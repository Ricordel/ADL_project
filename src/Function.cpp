#include <sstream>
#include "Function.hpp"


Function::Function(uint32_t nVariables) : m_nVariables(nVariables) {}

/* We need a virtual destructor, eventhough it does nothing */
Function::~Function() {}




/**********************************************************************************
 ********************* Functions of the form x0 + a + b + c.d *********************
 **********************************************************************************/

Function_0_a_b_cd::Function_0_a_b_cd()
        : Function(0), m_a(0), m_b(0), m_c(0), m_d(0)
{}

//XXX y a un truc avec les références ici, ou un truc du genre, en
//XXX rapport avec le passage direct de Function(a, b, c, d) dans
//XXX lower_or_equal().
Function_0_a_b_cd::Function_0_a_b_cd(Function_0_a_b_cd& other)
        : Function(other.m_nVariables),
          m_a(other.m_a), m_b(other.m_b), m_c(other.m_c), m_d(other.m_d)
{}


Function_0_a_b_cd::Function_0_a_b_cd(const Function_0_a_b_cd& other)
        : Function(other.m_nVariables),
          m_a(other.m_a), m_b(other.m_b), m_c(other.m_c), m_d(other.m_d)
{}


Function_0_a_b_cd::Function_0_a_b_cd(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t nVariables)
        : Function(nVariables), m_a(a), m_b(b), m_c(c), m_d(d)
{}

Function_0_a_b_cd::~Function_0_a_b_cd() {}




std::string Function_0_a_b_cd::toString() const
{
        std::ostringstream sstr;
        sstr << 0 << "," << m_a  << "," << m_b << ",(" << m_c << "," << m_d << ")";

        return sstr.str();
}


std::string Function_0_a_b_cd::toPrettyString() const
{
        std::stringstream sstr;
        sstr << "x_" << 0 << " + " << "x_" << m_a  << " + " << "x_" << m_b << " + "
             << "x_" <<  m_c << "." << "x_" <<  m_d;

        return sstr.str();
}



/*
 * "Equivalence" (meaning that one can be deducted from another) can appear
 * under several forms:
 *      - commutativity of + and . resulting in for instance
 *              x0 + x1 + x3 + x5.x6
 *        and
 *              x0 + x3 + x1 + x6.x5
 *        to be equivalent
 *
 *      - reverse functions having the same cycle length (see paper), for instance
 *              x0 + x1 + x3 + x5.x6
 *        and
 *              x0 + x6 + x5 + x3.x1
 *
 *      - any combination of the two
 *
 * A function a, b, (c, d) is considered ad being in canonical form if it's the
 * smallest of the following functions:
 *      a, b, (d, c) (commutativity)
 *      b, a, (c, d) (commutativity)
 *      b, a, (d, c) (commutativity)
 *
 *      d, c, (b, a) (reverse)
 *      d, c, (a, b) (reverse and commutativity)
 *      c, d, (b, a) (reverse and commutativity)
 *      c, d, (a, b) (reverse and commutativity)
 *
 */
inline bool Function_0_a_b_cd::isCanonicalForm() const
{
        // NOTE: number of variables is not usefull here, put to 0
        return (smaller_or_equal(Function_0_a_b_cd(m_a, m_b, m_d, m_c, 0))
             && smaller_or_equal(Function_0_a_b_cd(m_b, m_a, m_c, m_d, 0))
             && smaller_or_equal(Function_0_a_b_cd(m_b, m_a, m_d, m_c, 0))

             && smaller_or_equal(Function_0_a_b_cd(m_d, m_c, m_b, m_a, 0))
             && smaller_or_equal(Function_0_a_b_cd(m_d, m_c, m_a, m_b, 0))
             && smaller_or_equal(Function_0_a_b_cd(m_c, m_d, m_b, m_a, 0))
             && smaller_or_equal(Function_0_a_b_cd(m_c, m_d, m_a, m_b, 0)));
}


bool Function_0_a_b_cd::smaller_or_equal(Function_0_a_b_cd other) const
{
        if (m_a < other.m_a)
                return true;

        if (m_a == other.m_a && m_b < other.m_b)
                return true;

        if (m_a == other.m_a && m_b == other.m_b && m_c < other.m_c)
                return true;

        if (m_a == other.m_a && m_b == other.m_b && m_c == other.m_c && m_d <= other.m_d)
                return true;

        return false;
}
