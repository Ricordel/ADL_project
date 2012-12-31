#include <sstream>
#include <string>
#include <omp.h>

#ifdef WITHOUT_CPP11
#include <cstdlib>
#endif

#include "Function.hpp"


/* Switch for old fashioned g++ that does not support C++11 */
#ifndef WITHOUT_CPP11

#define PARSE_NUM(__var, __name, __form) \
do { \
        try { \
                __var = std::stoi(strRepr.substr(curPos), &substrPos); \
                curPos += substrPos; \
        } catch (std::exception& e) { \
                throw std::runtime_error( \
                                "Failed to parse " + strRepr + " as a function " + __form + ":" \
                                "cannot parse " + __name); \
        } \
} while (0)


#define EAT(__c, __after) \
do { \
        if (strRepr[curPos++] != __c) { \
                throw std::runtime_error( \
                                "Failed to parse " + strRepr + "there " \
                                "should be a '" + __c + "' after '" + __after + "'"); \
        } \
} while (0)


#endif /* ifndef WITHOUT_CPP11 */


Function::Function(uint32_t nVariables) : m_nVariables(nVariables) {}

/* We need a virtual destructor, eventhough it does nothing */
Function::~Function() {}




/**********************************************************************************
 ********************* Functions of the form x0 + a + b + c.d *********************
 **********************************************************************************/




Function_0_a_b_cd::Function_0_a_b_cd(int32_t a, int32_t b, int32_t c, int32_t d, uint32_t nVariables)
        : Function(nVariables), m_a(a), m_b(b), m_c(c), m_d(d)
{}


#ifndef WITHOUT_CPP11

Function_0_a_b_cd::Function_0_a_b_cd(const std::string& strRepr, uint32_t nVariables)
        throw (std::runtime_error) : Function(nVariables)
{
        // Would be nice with regex, but gcc does not support std::regex yet...
        // So we'll do some nasty stuff instead.

        size_t curPos = 0;
        size_t substrPos = 1234567;

        EAT('0', "start");
        EAT(',', "0");
        PARSE_NUM(m_a, "a", "0 + a + b + c.d");
        EAT(',', "a");
        PARSE_NUM(m_b, "b", "0 + a + b + c.d");
        EAT(',', "b");
        EAT('(', "b,");
        PARSE_NUM(m_c, "c", "0 + a + b + c.d");
        EAT(',', "c");
        PARSE_NUM(m_d, "d", "0 + a + b + c.d");
        EAT(')', "d");

        // Check it's the end
        if (curPos != strRepr.length()) {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b + c.d: "
                                "shouldn't be anything after ')'");
        }
}

#endif /* ifndef WITHOUT_CPP11 */


Function_0_a_b_cd::Function_0_a_b_cd()
        : Function(0), m_a(0), m_b(0), m_c(0), m_d(0) {}

Function_0_a_b_cd::Function_0_a_b_cd(Function_0_a_b_cd& other)
        : Function(other.m_nVariables),
          m_a(other.m_a), m_b(other.m_b), m_c(other.m_c), m_d(other.m_d)
{}

Function_0_a_b_cd::Function_0_a_b_cd(const Function_0_a_b_cd& other)
        : Function(other.m_nVariables),
          m_a(other.m_a), m_b(other.m_b), m_c(other.m_c), m_d(other.m_d)
{}

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
 *      - reverse functions having the same cycle length (see paper)
 *
 * Base function:
 *      x0 + xa + xb + xc.xd
 *
 * Reverse function:
 *      x0 + xar + xbr + xcr.xdr
 * where we define
 *      xar = N - xa, xbr = N - xb, xcr = N - xc, xdr = N - xd
 *
 * We know by construction that:
 *      xa < xb
 *      xc < xd
 *
 * Hence we can deduce that:
 *      xbr < xar
 *      xdr < xcr
 *      
 * So the "minimal" version of the reverse function is
 *      x0 + xbr + xar + xdr.xcr
 *
 * This is the only one we need to test against, the other ones being
 * "greater"
 *      
 */
inline bool Function_0_a_b_cd::isCanonicalForm() const
{
        int32_t ar = m_nVariables - m_a;
        int32_t br = m_nVariables - m_b;
        int32_t cr = m_nVariables - m_c;
        int32_t dr = m_nVariables - m_d;

        return smaller_or_equal(Function_0_a_b_cd(br, ar, dr, cr, m_nVariables));
}


inline bool Function_0_a_b_cd::smaller_or_equal(Function_0_a_b_cd other) const
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


Function_0_a_bc_de::Function_0_a_bc_de(int32_t a, int32_t b, int32_t c,
                                       int32_t d, int32_t e, uint32_t nVariables)
        : Function(nVariables), m_a(a), m_b(b), m_c(c), m_d(d), m_e(e)
{}


#ifndef WITHOUT_CPP11

Function_0_a_bc_de::Function_0_a_bc_de(const std::string& strRepr, uint32_t nVariables)
        throw (std::runtime_error) : Function(nVariables)
{
        // Would be nice with regex, but gcc does not support std::regex yet...
        // So we'll do some nasty stuff instead.

        size_t curPos = 0;
        size_t substrPos = 1234567;

        EAT('0', "start");
        EAT(',', "0");

        PARSE_NUM(m_a, "a", "0 + a + b.c + d.e");

        EAT(',', "a");
        EAT('(', "a,");

        PARSE_NUM(m_b, "b", "0 + a + b.c + d.e");

        EAT(',', "b");

        PARSE_NUM(m_c, "c", "0 + a + b.c + d.e");

        EAT(')', "c");
        EAT(',', "c)");
        EAT('(', "c(,");

        PARSE_NUM(m_d, "d", "0 + a + b.c + d.e");

        EAT(',', "d");

        PARSE_NUM(m_e, "e", "0 + a + b.c + d.e");

        EAT(')', "e");

        // Check it's the end
        if (curPos != strRepr.length()) {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b.c + d.e: "
                                "shouldn't be anything after ')'");
        }
}

#endif /* ifndef WITHOUT_CPP11 */


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
 * We know by construction that (see generation code):
 *      xb < xc
 *      xd < xe
 *      xb <= xd
 *
 * So we can deduce:
 *      xcr < xbr
 *      xer < xdr
 *      xdr <= xbr
 *
 * So the smallest reverse function is either
 *      x0 + xar + xer.xdr + xcr.xbr
 * or
 *      x0 + xar + xcr.xbr + xer.xdr
 *
 * We also need to test against
 *      x0 + xa + xd.xe + xb.xc
 * which is not taken care of in the generation process
 *
 * where we define:
 *      xar = N-a, xbr = N-b, xcr = N-c, xdr = N-d, xer = N-e :
 *
 * Last thing, we don't want functions of the form
 *      a + b.c + b.c
 * which are "not really" of the a_bc_de form
 */
inline bool Function_0_a_bc_de::isCanonicalForm() const
{
        int32_t ar = m_nVariables - m_a;
        int32_t br = m_nVariables - m_b;
        int32_t cr = m_nVariables - m_c;
        int32_t dr = m_nVariables - m_d;
        int32_t er = m_nVariables - m_e;

        if (m_b == m_d && m_c == m_e) {
                return false;
        }

        return (smaller_or_equal(Function_0_a_bc_de(ar, er, dr, cr, br, m_nVariables))
             && smaller_or_equal(Function_0_a_bc_de(ar, cr, br, er, dr, m_nVariables))
             && smaller_or_equal(Function_0_a_bc_de(m_a, m_d, m_e, m_b, m_c, m_nVariables)));
}



inline bool Function_0_a_bc_de::smaller_or_equal(Function_0_a_bc_de other) const
{
        if (m_a < other.m_a)
                return true;

        if (m_a == other.m_a && m_b < other.m_b)
                return true;

        if (m_a == other.m_a && m_b == other.m_b && m_c < other.m_c)
                return true;

        if (m_a == other.m_a && m_b == other.m_b && m_c == other.m_c && m_d < other.m_d)
                return true;

        if (m_a == other.m_a && m_b == other.m_b && m_c == other.m_c && m_d == other.m_d && m_e <= other.m_e)
                return true;

        return false;
}







/***********************************************************************
 **************** For x0 + xa + xb + xc + xd + xe.xf *******************
 ***********************************************************************/


Function_0_a_b_c_d_ef::Function_0_a_b_c_d_ef()
        : Function(0), m_a(0), m_b(0), m_c(0), m_d(0), m_e(0), m_f(0)
{}


Function_0_a_b_c_d_ef::Function_0_a_b_c_d_ef(int32_t a, int32_t b, int32_t c, int32_t d,
                                             int32_t e, int32_t f, uint32_t nVariables)
        : Function(nVariables), m_a(a), m_b(b), m_c(c), m_d(d), m_e(e), m_f(f)
{}


Function_0_a_b_c_d_ef::~Function_0_a_b_c_d_ef() {}




std::string Function_0_a_b_c_d_ef::toString() const
{
        std::ostringstream sstr;
        sstr << m_nVariables << " variables: "
             << 0 << "," << m_a  << "," << m_b << "," << m_c << "," << m_d << ",(" << m_e << "," << m_f << ")";

        return sstr.str();
}


std::string Function_0_a_b_c_d_ef::toPrettyString() const
{
        std::stringstream sstr;
        sstr << "x_" << 0 << " + " << "x_" << m_a  << " + "
             << "x_" << m_b << " + " << "x_" <<  m_c << " + " << "x_" << m_d << " + "
             << "(x_" << m_e << "." << "x_" << m_f << ")";

        return sstr.str();
}



/*
 * Considering that the base version verifies
 *      a < b < c < d
 *      e < f
 * the smallest version of the reverse function is:
 *      dr + cr + br + ar + fr.er
 *
 */
inline bool Function_0_a_b_c_d_ef::isCanonicalForm() const
{
        int32_t ar = m_nVariables - m_a;
        int32_t br = m_nVariables - m_b;
        int32_t cr = m_nVariables - m_c;
        int32_t dr = m_nVariables - m_d;
        int32_t er = m_nVariables - m_e;
        int32_t fr = m_nVariables - m_f;

        return smaller_or_equal(Function_0_a_b_c_d_ef(dr, cr, br, ar, fr, er, m_nVariables));
}



inline bool Function_0_a_b_c_d_ef::smaller_or_equal(Function_0_a_b_c_d_ef other) const
{
        if (m_a < other.m_a)
                return true;

        if (m_a == other.m_a && m_b < other.m_b)
                return true;

        if (m_a == other.m_a && m_b == other.m_b && m_c < other.m_c)
                return true;

        if (m_a == other.m_a && m_b == other.m_b && m_c == other.m_c && m_d < other.m_d)
                return true;

        if (m_a == other.m_a && m_b == other.m_b && m_c == other.m_c && m_d == other.m_d && m_e < other.m_e)
                return true;

        if (m_a == other.m_a && m_b == other.m_b && m_c == other.m_c
         && m_d == other.m_d && m_e == other.m_e && m_f <= other.m_f)
                return true;

        return false;
}
                

#ifndef WITHOUT_CPP11

Function_0_a_b_c_d_ef::Function_0_a_b_c_d_ef(const std::string& strRepr, uint32_t nVariables)
        throw (std::runtime_error)
        : Function(nVariables)
{
        size_t curPos = 0;
        size_t substrPos = 1234567;

        EAT('0', "start");
        EAT(',', "0");
        PARSE_NUM(m_a, "a", "0 + a + b + c + d + e.f");
        EAT(',', "a");
        PARSE_NUM(m_b, "b", "0 + a + b + c + d + e.f");
        EAT(',', "b");
        PARSE_NUM(m_c, "c", "0 + a + b + c + d + e.f");
        EAT(',', "c");
        PARSE_NUM(m_d, "d", "0 + a + b + c + d + e.f");
        EAT(',', "d");
        EAT('(', "d,");
        PARSE_NUM(m_e, "e", "0 + a + b + c + d + e.f");
        EAT(',', "e");
        PARSE_NUM(m_f, "f", "0 + a + b + c + d + e.f");
        EAT(')', "f");

        // Check it's the end
        if (curPos != strRepr.length()) {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b + c + d + e.f: "
                                "shouldn't be anything after ')'");
        }
}

#endif /* ifndef WITHOUT_CPP11 */



/***********************************************************************
 ******************** For x0 + xa + xb + xc.xd.xe **********************
 ***********************************************************************/


Function_0_a_b_cde::Function_0_a_b_cde()
        : Function(0), m_a(0), m_b(0), m_c(0), m_d(0), m_e(0)
{}


Function_0_a_b_cde::Function_0_a_b_cde(int32_t a, int32_t b, int32_t c,
                                       int32_t d, int32_t e, uint32_t nVariables)
        : Function(nVariables), m_a(a), m_b(b), m_c(c), m_d(d), m_e(e)
{}


Function_0_a_b_cde::~Function_0_a_b_cde() {}




std::string Function_0_a_b_cde::toString() const
{
        std::ostringstream sstr;
        sstr << m_nVariables << " variables: "
             << 0 << "," << m_a  << "," << m_b << ",(" << m_c << "," << m_d << "," << m_e << ")";

        return sstr.str();
}


std::string Function_0_a_b_cde::toPrettyString() const
{
        std::stringstream sstr;
        sstr << "x_" << 0 << " + " << "x_" << m_a  << " + " << "x_" << m_b << " + "
             << "x_" <<  m_c << "." << "x_" << m_d << "." << "x_" << m_e;

        return sstr.str();
}



/*
 * Considering that the base version verifies
 *      a < b
 *      c < d < e
 * the smallest version of the reverse function is:
 *      br + ar + er.dr.cr
 *
 */
inline bool Function_0_a_b_cde::isCanonicalForm() const
{
        int32_t ar = m_nVariables - m_a;
        int32_t br = m_nVariables - m_b;
        int32_t cr = m_nVariables - m_c;
        int32_t dr = m_nVariables - m_d;
        int32_t er = m_nVariables - m_e;

        return smaller_or_equal(Function_0_a_b_cde(br, ar, er, dr, cr, m_nVariables));
}



inline bool Function_0_a_b_cde::smaller_or_equal(Function_0_a_b_cde other) const
{
        if (m_a < other.m_a)
                return true;

        if (m_a == other.m_a && m_b < other.m_b)
                return true;

        if (m_a == other.m_a && m_b == other.m_b && m_c < other.m_c)
                return true;

        if (m_a == other.m_a && m_b == other.m_b && m_c == other.m_c && m_d < other.m_d)
                return true;

        if (m_a == other.m_a && m_b == other.m_b && m_c == other.m_c && m_d == other.m_d && m_e <= other.m_e)
                return true;

        return false;
}
                



#ifndef WITHOUT_CPP11

Function_0_a_b_cde::Function_0_a_b_cde(const std::string& strRepr, uint32_t nVariables)
        throw (std::runtime_error)
        : Function(nVariables)
{
        size_t curPos = 0;
        size_t substrPos = 1234567;


        EAT('0', "start");
        EAT(',', "0");
        PARSE_NUM(m_a, "a", "0 + a + b + c + d + e.f");
        EAT(',', "a");
        PARSE_NUM(m_b, "b", "0 + a + b + c + d + e.f");
        EAT(',', "b");
        EAT('(', "b,");
        PARSE_NUM(m_c, "c", "0 + a + b + c + d + e.f");
        EAT(',', "c");
        PARSE_NUM(m_d, "d", "0 + a + b + c + d + e.f");
        EAT(',', "d");
        PARSE_NUM(m_e, "e", "0 + a + b + c + d + e.f");
        EAT(')', "e");

        // Check it's the end
        if (curPos != strRepr.length()) {
                throw std::runtime_error(
                                "Failed to parse " + strRepr + " as a function 0 + a + b + c.d.e: "
                                "shouldn't be anything after ')'");
        }
}

#endif /* ifndef WITHOUT_CPP11 */