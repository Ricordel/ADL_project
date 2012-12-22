#ifndef __FUNCTION_H__
#define __FUNCTION_H__

#include <string>
#include <iostream>



class Function
{
        public:
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
                virtual bool isCanonicalForm() const = 0;

                /* Return the length of the NLFSR cycle for this particular function */
                inline uint32_t getCycleLength() const {
                        int length = 0;
                        while (this->nextVal() != startVal) {
                                length++;
                        }

                        return length;
                }

        protected:
                uint32_t curVal;
                uint32_t startVal = 1;
                
                /* Side effects on the curVal member AND returns its value */
                virtual uint32_t nextVal() const = 0;
};




class Function_0_a_b_cd : public Function
{
        public:
                Function_0_a_b_cd();
                Function_0_a_b_cd(Function_0_a_b_cd& other);
                Function_0_a_b_cd(const Function_0_a_b_cd& other);
                Function_0_a_b_cd(uint32_t a, uint32_t b, uint32_t c, uint32_t d);
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
                virtual inline uint32_t nextVal() const {
                        std::cout << "Mettre la génération à la Elena ici" << std::endl;
                        return 0;
                }

                friend class FuncGenerator_0_a_b_cd;
};


#endif /* end of include guard: __FUNCTION_H__ */
