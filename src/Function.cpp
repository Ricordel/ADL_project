#include <sstream>
#include <string>

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


Function_0_a_b_cd::Function_0_a_b_cd(const std::string& strRepr, uint32_t nVariables)
        throw (std::runtime_error) : Function(nVariables)
{
        // Would be nice with regex, but gcc does not support std::regex yet...
        // So we'll do some nasty stuff instead.

        size_t curPos = 0;
        size_t substrPos = 1234567;

        if (strRepr[curPos++] != '0' || strRepr[curPos++] != ',') {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b + c.d: "
                                "does not start with '0,'");
        }


        // Parse a number
        try {
                m_a = std::stoi(strRepr.substr(curPos), &substrPos);
                curPos += substrPos;
        } catch (std::exception& e) {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b + c.d: "
                                "cannot parse 'a'");
        }

        // Check next is a comma
        if (strRepr[curPos++] != ',') {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b + c.d: "
                                "should be a ',' after 'a'");
        }

        // Parse b
        try {
                m_b = std::stoi(strRepr.substr(curPos), &substrPos);
                curPos += substrPos;
        } catch (std::exception& e) {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b + c.d: "
                                "cannot parse 'b'");
        }

        // Check next is a comma and opening parenthese
        if (strRepr[curPos++] != ',' || strRepr[curPos++] != '(') {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b + c.d: "
                                "should be ',(' after 'b'");
        }

        // parse c
        try {
                m_c = std::stoi(strRepr.substr(curPos), &substrPos);
                curPos += substrPos;
        } catch (std::exception& e) {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b + c.d: "
                                "cannot parse 'c'");
        }

        // Check next is a comma
        if (strRepr[curPos++] != ',') {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b + c.d: "
                                "should be ',' after 'c'");
        }

        // parse d
        try {
                m_d = std::stoi(strRepr.substr(curPos), &substrPos);
                curPos += substrPos;
        } catch (std::exception& e) {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b + c.d: "
                                "cannot parse 'd'");
        }

        // Check next is a closing parenthese
        if (strRepr[curPos++] != ')') {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b + c.d: "
                                "should be ')' after 'd'");
        }

        // Check it's the end
        if (curPos != strRepr.length()) {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b + c.d: "
                                "shouldn't be anything after ')'");
        }
}


Function_0_a_b_cd::~Function_0_a_b_cd() {}




std::string Function_0_a_b_cd::toString() const
{
        std::ostringstream sstr;
        sstr << m_nVariables << " variables: "
             << 0 << "," << m_a  << "," << m_b << ",(" << m_c << "," << m_d << ")";

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






/**********************************************************************************
 ******************** Functions of the form x0 + a + b.c + d.e ********************
 **********************************************************************************/

Function_0_a_bc_de::Function_0_a_bc_de()
        : Function(0), m_a(0), m_b(0), m_c(0), m_d(0), m_e(0)
{}


Function_0_a_bc_de::Function_0_a_bc_de(Function_0_a_bc_de& other)
        : Function(other.m_nVariables),
          m_a(other.m_a), m_b(other.m_b), m_c(other.m_c), m_d(other.m_d), m_e(other.m_e)
{}


Function_0_a_bc_de::Function_0_a_bc_de(const Function_0_a_bc_de& other)
        : Function(other.m_nVariables),
          m_a(other.m_a), m_b(other.m_b), m_c(other.m_c), m_d(other.m_d), m_e(other.m_e)
{}


Function_0_a_bc_de::Function_0_a_bc_de(uint32_t a, uint32_t b, uint32_t c,
                                       uint32_t d, uint32_t e, uint32_t nVariables)
        : Function(nVariables), m_a(a), m_b(b), m_c(c), m_d(d), m_e(e)
{}


Function_0_a_bc_de::Function_0_a_bc_de(const std::string& strRepr, uint32_t nVariables)
        throw (std::runtime_error) : Function(nVariables)
{
        // Would be nice with regex, but gcc does not support std::regex yet...
        // So we'll do some nasty stuff instead.

        size_t curPos = 0;
        size_t substrPos = 1234567;

        if (strRepr[curPos++] != '0' || strRepr[curPos++] != ',') {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b.c + d.e: "
                                "does not start with '0,'");
        }


        // Parse a
        try {
                m_a = std::stoi(strRepr.substr(curPos), &substrPos);
                curPos += substrPos;
        } catch (std::exception& e) {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b.c + d.e: "
                                "cannot parse 'a'");
        }

        if (strRepr[curPos++] != ',' || strRepr[curPos++] != '(') {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b.c + d.e: "
                                "should be a ',(' after 'a'");
        }

        // Parse b
        try {
                m_b = std::stoi(strRepr.substr(curPos), &substrPos);
                curPos += substrPos;
        } catch (std::exception& e) {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b.c + d.e: "
                                "cannot parse 'b'");
        }

        if (strRepr[curPos++] != ',') {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b.c + d.e: "
                                "should be ',' after 'b'");
        }

        // parse c
        try {
                m_c = std::stoi(strRepr.substr(curPos), &substrPos);
                curPos += substrPos;
        } catch (std::exception& e) {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b.c + d.e: "
                                "cannot parse 'c'");
        }

        if (strRepr[curPos++] != ')' || strRepr[curPos++] != ',' || strRepr[curPos++] != '(') {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b.c + d.e: "
                                "should be '),(' after 'c'");
        }


        // parse d
        try {
                m_d = std::stoi(strRepr.substr(curPos), &substrPos);
                curPos += substrPos;
        } catch (std::exception& e) {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b.c + d.e: "
                                "cannot parse 'd'");
        }

        if (strRepr[curPos++] != ',') {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b.c + d.e: "
                                "should be ',' after 'd'");
        }

        // parse e
        try {
                m_e = std::stoi(strRepr.substr(curPos), &substrPos);
                curPos += substrPos;
        } catch (std::exception& e) {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b.c + d.e: "
                                "cannot parse 'e'");
        }

        if (strRepr[curPos++] != ')') {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b.c + d.e: "
                                "should be ')' after 'e'");
        }

        // Check it's the end
        if (curPos != strRepr.length()) {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b.c + d.e: "
                                "shouldn't be anything after ')'");
        }
}


Function_0_a_bc_de::~Function_0_a_bc_de() {}




std::string Function_0_a_bc_de::toString() const
{
        std::ostringstream sstr;
        sstr << m_nVariables << " variables: "
             << 0 << "," << m_a  << ",(" << m_b << "," << m_c << "),(" << m_d << "," << m_e << ")";

        return sstr.str();
}


std::string Function_0_a_bc_de::toPrettyString() const
{
        std::stringstream sstr;
        sstr << "x_" << 0 << " + " << "x_" << m_a  << " + "
             << "(x_" << m_b << "." << "x_" <<  m_c << ") + "
             << "(x_" << m_d << "." << "x_" << m_e << ")";

        return sstr.str();
}



/*
 * As for the first case, duplication comes from reverse functions
 * and commuttivity of . and +. For this form, the duplications are
 * even more numerous :
 *
 * x0 + xa + xb.xc + xd.xe
 *
 * x0 + xa + xc.xb + xd.xe (commutativity of .)
 * x0 + xa + xb.xc + xe.xd (commutativity of .)
 * x0 + xa + xc.xb + xe.xd (commutativity of .)
 *
 * x0 + xa + xd.xe + xb.xc (commutativity of +)
 * x0 + xa + xe.xd + xb.xc (commutativity of + and then .)
 * x0 + xa + xd.xe + xc.xb (commutativity of + and then .)
 * x0 + xa + xe.xd + xc.xb (commutativity of + and then .)
 *
 * x0 + xe + xd.xc + xb.xa (reverse function)
 *
 * x0 + xe + xc.xd + xb.xa (reverse function and commutativity of .)
 * x0 + xe + xd.xc + xa.xb (reverse function and commutativity of .)
 * x0 + xe + xc.xd + xa.xb (reverse function and commutativity of .)
 *
 * x0 + xe + xb.xa + xd.xc (reverse function and commutativity of . and then +)
 * x0 + xe + xa.xb + xd.xc (reverse function and commutativity of . and then +)
 * x0 + xe + xb.xa + xc.xd (reverse function and commutativity of . and then +)
 * x0 + xe + xa.xb + xc.xd (reverse function and commutativity of . and then +)
 *
 *
 * The same way as for the first form of functions, we consider a function to be
 * in canonical form if it's the smallest (in lexicographical order) of all
 * the variants enumerated above.
 */
inline bool Function_0_a_bc_de::isCanonicalForm() const
{
        // NOTE: number of variables is not usefull here, put to 0
        return (smaller_or_equal(Function_0_a_bc_de(m_a, m_c, m_b, m_d, m_e, 0))
             && smaller_or_equal(Function_0_a_bc_de(m_a, m_b, m_c, m_e, m_d, 0))
             && smaller_or_equal(Function_0_a_bc_de(m_a, m_c, m_b, m_e, m_d, 0))

             && smaller_or_equal(Function_0_a_bc_de(m_a, m_d, m_e, m_b, m_c, 0))
             && smaller_or_equal(Function_0_a_bc_de(m_a, m_e, m_d, m_b, m_c, 0))
             && smaller_or_equal(Function_0_a_bc_de(m_a, m_d, m_e, m_c, m_b, 0))
             && smaller_or_equal(Function_0_a_bc_de(m_a, m_e, m_d, m_c, m_b, 0))

             && smaller_or_equal(Function_0_a_bc_de(m_e, m_d, m_c, m_b, m_a, 0))

             && smaller_or_equal(Function_0_a_bc_de(m_e, m_c, m_d, m_b, m_a, 0))
             && smaller_or_equal(Function_0_a_bc_de(m_e, m_d, m_c, m_a, m_b, 0))
             && smaller_or_equal(Function_0_a_bc_de(m_e, m_c, m_d, m_a, m_b, 0))

             && smaller_or_equal(Function_0_a_bc_de(m_e, m_b, m_a, m_d, m_c, 0))
             && smaller_or_equal(Function_0_a_bc_de(m_e, m_a, m_b, m_d, m_c, 0))
             && smaller_or_equal(Function_0_a_bc_de(m_e, m_b, m_a, m_c, m_d, 0))
             && smaller_or_equal(Function_0_a_bc_de(m_e, m_a, m_b, m_c, m_d, 0)));
}



bool Function_0_a_bc_de::smaller_or_equal(Function_0_a_bc_de other) const
{
        if (m_a < other.m_a)
                return true;

        if (m_a == other.m_a && m_b < other.m_b)
                return true;

        if (m_a == other.m_a && m_b == other.m_b && m_c < other.m_c)
                return true;

        if (m_a == other.m_a && m_b == other.m_b && m_c == other.m_c && m_d < other.m_d)
                return true;

        if (m_a == other.m_a && m_b == other.m_b && m_c == other.m_c && m_d < other.m_d && m_e < other.m_e)
                return true;

        return false;
}
