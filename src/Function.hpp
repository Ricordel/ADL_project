#ifndef __FUNCTION_H__
#define __FUNCTION_H__

#include <string>
#include <iostream>
#include <stdexcept>


/* 
 * Return the bit nBit of val in first position. That is, the return
 * value of this function is either 0 or 1.
 */
static inline uint32_t bit(uint32_t nBit, uint32_t val)
{
        return (val >> nBit) & 1;
}


class Function
{
        public:
                Function(uint32_t nVariables);
                virtual ~Function();

                /* This produces the canonical form required for the project,
                 * for example 1,2,(5,6) */
                virtual std::string toString() const = 0;

                /* This produces a nicely readable string representation of the
                 * functions, for example x_1 + x_2 + x_5.x_6 */
                virtual std::string toPrettyString() const = 0;

                /* Tell if the function is in canonical form, that is if there is
                 * not a "smaller" function that is equivalent thanks to commutativity
                 * or "reverse equivalence" */
                virtual inline bool isCanonicalForm() const = 0;

                /* Return the length of the NLFSR cycle for this particular function */
                inline uint32_t getCycleLength()
                {
                        int length = 1;
                        m_curVal = 1;
                        while (this->nextVal() != 1) { /* back to start value */
                                length++;
                        }

                        return length;
                }

                inline void printCycle(std::ostream& outStream)
                {
                        m_curVal = 1;
                        do {
                                outStream << m_curVal << " ";
                        } while (this->nextVal() != 1);
                }

                inline void printDetails(std::ostream& outStream)
                {
                        outStream << toPrettyString() << std::endl << "\t";
                        printCycle(outStream);
                        outStream << std::endl;
                }

        protected:
                uint32_t m_curVal;
                uint32_t m_nVariables;
                
                /* Side effects on the m_curVal member AND returns its value */
                virtual uint32_t nextVal() = 0;
};



class Function_0_a_b_cd : public Function
{
        public:
                Function_0_a_b_cd();
                Function_0_a_b_cd(Function_0_a_b_cd& other);
                Function_0_a_b_cd(const Function_0_a_b_cd& other);
                Function_0_a_b_cd(int32_t a, int32_t b, int32_t c, int32_t d, uint32_t m_nVariables);
                Function_0_a_b_cd(const std::string& strRepr, uint32_t nVariables) throw (std::runtime_error);
                virtual ~Function_0_a_b_cd();

                virtual std::string toString() const;
                virtual std::string toPrettyString() const;
                virtual bool isCanonicalForm() const;

        private:
                int32_t m_a;
                int32_t m_b;
                int32_t m_c;
                int32_t m_d;
                bool smaller_or_equal(Function_0_a_b_cd other) const;

        protected:
                virtual inline uint32_t nextVal() {
                        uint32_t newBit = bit(0, m_curVal) ^ bit(m_a, m_curVal) ^ bit(m_b, m_curVal) ^
                                         (bit(m_c, m_curVal) & bit(m_d, m_curVal));

                        m_curVal = (m_curVal >> 1) | (newBit << (m_nVariables - 1));
                        
                        return m_curVal;
                }


                friend class FuncGenerator_0_a_b_cd;
};





class Function_0_a_bc_de : public Function
{
        public:
                Function_0_a_bc_de();
                Function_0_a_bc_de(Function_0_a_bc_de& other);
                Function_0_a_bc_de(const Function_0_a_bc_de& other);
                Function_0_a_bc_de(int32_t a, int32_t b, int32_t c, int32_t d, int32_t e, uint32_t m_nVariables);
                Function_0_a_bc_de(const std::string& strRepr, uint32_t nVariables) throw (std::runtime_error);
                virtual ~Function_0_a_bc_de();

                virtual std::string toString() const;
                virtual std::string toPrettyString() const;
                virtual bool isCanonicalForm() const;

        private:
                int32_t m_a;
                int32_t m_b;
                int32_t m_c;
                int32_t m_d;
                int32_t m_e;
                bool smaller_or_equal(Function_0_a_bc_de other) const;

        protected:
                virtual inline uint32_t nextVal() {
                        uint32_t newBit = bit(0, m_curVal) ^ bit(m_a, m_curVal) ^
                                         (bit(m_b, m_curVal) & bit(m_c, m_curVal)) ^
                                         (bit(m_d, m_curVal) & bit(m_e, m_curVal));

                        m_curVal = (m_curVal >> 1) | (newBit << (m_nVariables - 1));
                        
                        return m_curVal;
                }


                friend class FuncGenerator_0_a_bc_de;
};


class Function_0_a_b_c_d_ef : public Function
{
        public:
                Function_0_a_b_c_d_ef();
                Function_0_a_b_c_d_ef(int32_t a, int32_t b, int32_t c, int32_t d,
                                      int32_t e, int32_t f, uint32_t m_nVariables);
                Function_0_a_b_c_d_ef(const std::string& strRepr, uint32_t nVariables) throw (std::runtime_error);
                virtual ~Function_0_a_b_c_d_ef();

                virtual std::string toString() const;
                virtual std::string toPrettyString() const;
                virtual bool isCanonicalForm() const;

        private:
                int32_t m_a;
                int32_t m_b;
                int32_t m_c;
                int32_t m_d;
                int32_t m_e;
                int32_t m_f;
                bool smaller_or_equal(Function_0_a_b_c_d_ef other) const;

        protected:
                virtual inline uint32_t nextVal() {
                        uint32_t newBit = bit(0, m_curVal) ^ bit(m_a, m_curVal) ^ bit(m_b, m_curVal) ^
                                          bit(m_c, m_curVal) ^ bit(m_d, m_curVal) ^
                                          (bit(m_e, m_curVal) & bit(m_f, m_curVal));

                        m_curVal = (m_curVal >> 1) | (newBit << (m_nVariables - 1));
                        
                        return m_curVal;
                }


                friend class FuncGenerator_0_a_b_c_d_ef;
};




class Function_0_a_b_cde : public Function
{
        public:
                Function_0_a_b_cde();
                Function_0_a_b_cde(int32_t a, int32_t b, int32_t c,
                                   int32_t d, int32_t e, uint32_t m_nVariables);
                Function_0_a_b_cde(const std::string& strRepr, uint32_t nVariables) throw (std::runtime_error);
                virtual ~Function_0_a_b_cde();

                virtual std::string toString() const;
                virtual std::string toPrettyString() const;
                virtual bool isCanonicalForm() const;

        private:
                int32_t m_a;
                int32_t m_b;
                int32_t m_c;
                int32_t m_d;
                int32_t m_e;
                bool smaller_or_equal(Function_0_a_b_cde other) const;

        protected:
                virtual inline uint32_t nextVal() {
                        uint32_t newBit = bit(0, m_curVal) ^ bit(m_a, m_curVal) ^ bit(m_b, m_curVal) ^
                                          (bit(m_c, m_curVal) & bit(m_d, m_curVal) & bit(m_e, m_curVal));

                        m_curVal = (m_curVal >> 1) | (newBit << (m_nVariables - 1));
                        
                        return m_curVal;
                }


                friend class FuncGenerator_0_a_b_cde;
};

#endif /* end of include guard: __FUNCTION_H__ */
