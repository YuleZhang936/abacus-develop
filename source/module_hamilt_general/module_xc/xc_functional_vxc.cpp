// This file contains interface to xc_functional class:
// 1. v_xc : which takes rho as input, and v_xc as output
// 2. v_xc_libxc : which does the same thing as v_xc, but calling libxc
// NOTE : it is only used for nspin = 1 and 2, the nspin = 4 case is treated in v_xc
// 3. v_xc_meta : which takes rho and tau as input, and v_xc as output

#include "xc_functional.h"
#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_parameter/parameter.h"

#ifdef USE_LIBXC
#include "NCLibxc/NCLibxc.h"
#include "xc_functional_libxc.h"
#include <tuple>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <xc.h>
#include <stdexcept>
#endif

// [etxc, vtxc, v] = XC_Functional::v_xc(...)
std::tuple<double,double,ModuleBase::matrix> XC_Functional::v_xc(
	const int &nrxx, // number of real-space grid
    const Charge* const chr,
    const UnitCell *ucell) // core charge density
{
    ModuleBase::TITLE("XC_Functional","v_xc");
    ModuleBase::timer::tick("XC_Functional","v_xc");

    if(use_libxc)
    {
#ifdef USE_LIBXC
        return XC_Functional_Libxc::v_xc_libxc(XC_Functional::get_func_id(), nrxx, ucell->omega, ucell->tpiba, chr);
#else
        ModuleBase::WARNING_QUIT("v_xc","compile with LIBXC");
#endif
    }

    //Exchange-Correlation potential Vxc(r) from n(r)
    double etxc = 0.0;
    double vtxc = 0.0;
    ModuleBase::matrix v(PARAM.inp.nspin, nrxx);

    // the square of the e charge
    // in Rydeberg unit, so * 2.0.
    double e2 = 2.0;
    std::complex<double> twoi(0.0, 2.0);
    std::complex<double> two(2.0, 0.0);

    double vanishing_charge = 1.0e-10;

    if (PARAM.inp.nspin == 1 || ( PARAM.inp.nspin ==4 && !PARAM.globalv.domag && !PARAM.globalv.domag_z))
    {
        // spin-unpolarized case
#ifdef _OPENMP
#pragma omp parallel for reduction(+:etxc) reduction(+:vtxc)
#endif
        for (int ir = 0;ir < nrxx;ir++)
        {
            // total electron charge density
            double rhox = chr->rho[0][ir] + chr->rho_core[ir];
            if(PARAM.inp.use_paw && rhox < 1e-14) { rhox = 1e-14;
}
            double arhox = std::abs(rhox);
            if (arhox > vanishing_charge)
            {
                double exc = 0.0;
                double vxc = 0.0;
                XC_Functional::xc(arhox, exc, vxc);
                v(0,ir) = e2 * vxc;
                // consider the total charge density
                etxc += e2 * exc * rhox;
                // only consider chr->rho
                vtxc += v(0, ir) * chr->rho[0][ir];
            } // endif
        } //enddo
    }
    else if(PARAM.inp.nspin ==2)
    {
        // spin-polarized case
#ifdef _OPENMP
#pragma omp parallel for reduction(+:etxc) reduction(+:vtxc)
#endif
        for (int ir = 0;ir < nrxx;ir++)
        {
            double rhox = chr->rho[0][ir] + chr->rho[1][ir] + chr->rho_core[ir]; //HLX(05-29-06): bug fixed
            double arhox = std::abs(rhox);

            if (arhox > vanishing_charge)
            {
                double zeta = (chr->rho[0][ir] - chr->rho[1][ir]) / arhox; //HLX(05-29-06): bug fixed

                if (std::abs(zeta)  > 1.0)
                {
                    zeta = (zeta > 0.0) ? 1.0 : (-1.0);
                }

                double rhoup = arhox * (1.0+zeta) / 2.0;
                double rhodw = arhox * (1.0-zeta) / 2.0;
                double exc = 0.0;
                double vxc[2];
                XC_Functional::xc_spin(arhox, zeta, exc, vxc[0], vxc[1]);

                for (int is = 0;is < PARAM.inp.nspin;is++)
                {
                    v(is, ir) = e2 * vxc[is];
                }

                etxc += e2 * exc * rhox;
                vtxc += v(0, ir) * chr->rho[0][ir] + v(1, ir) * chr->rho[1][ir];
            }
        }
    }
    else if(PARAM.inp.nspin == 4 && PARAM.inp.multicolin == 0)
    { //noncollinear case (Kubler's locally collinear) added by zhengdy
#ifdef _OPENMP
#pragma omp parallel for reduction(+:etxc) reduction(+:vtxc)
#endif
        for(int ir = 0;ir<nrxx; ir++)
        {
            double amag = sqrt( pow(chr->rho[1][ir],2) + pow(chr->rho[2][ir],2) + pow(chr->rho[3][ir],2) );

            double rhox = chr->rho[0][ir] + chr->rho_core[ir];

            double arhox = std::abs( rhox );

            if ( arhox > vanishing_charge )
            {
                double zeta = amag / arhox;
                double exc = 0.0;
                double vxc[2];

                if ( std::abs( zeta ) > 1.0 )
                {
                    zeta = (zeta > 0.0) ? 1.0 : (-1.0);
                }//end if

                if(use_libxc)
                {
#ifdef USE_LIBXC
                    double rhoup = arhox * (1.0+zeta) / 2.0;
                    double rhodw = arhox * (1.0-zeta) / 2.0;
                    XC_Functional_Libxc::xc_spin_libxc(XC_Functional::get_func_id(), rhoup, rhodw, exc, vxc[0], vxc[1]);
#else
                    ModuleBase::WARNING_QUIT("v_xc","compile with LIBXC");
#endif                    
                }
                else
                {
                    double rhoup = arhox * (1.0+zeta) / 2.0;
                    double rhodw = arhox * (1.0-zeta) / 2.0;
                    XC_Functional::xc_spin(arhox, zeta, exc, vxc[0], vxc[1]);
                }

                etxc += e2 * exc * rhox;

                v(0, ir) = e2*( 0.5 * ( vxc[0] + vxc[1]) );
                vtxc += v(0,ir) * chr->rho[0][ir];

                double vs = 0.5 * ( vxc[0] - vxc[1] );
                if ( amag > vanishing_charge )
                {
                    for(int ipol = 1;ipol< 4;ipol++)
                    {
                        v(ipol, ir) = e2 * vs * chr->rho[ipol][ir] / amag;
                        vtxc += v(ipol,ir) * chr->rho[ipol][ir];
                    }//end do
                }//end if
            }//end if
        }//end do
    }//end if
    // energy terms, local-density contributions

    if (PARAM.inp.nspin == 4 && PARAM.inp.multicolin == 1
        && (func_type == 0 || func_type == 1) )
    { // noncollinear case added by Xiaoyu Zhang, Peking University, 2024.10.02.  multicollinear method for lda Since NCLibxc needs libxc, this part codes will not be used.

        std::cerr << "Error: Multi-collinear approach does not support running without Libxc." << std::endl;
        std::exit(EXIT_FAILURE);
//        NCLibxc::print_NCLibxc();
//#ifdef _OPENMP
//#pragma omp parallel for reduction(+:etxc) reduction(+:vtxc)
//#endif
//        for(int ir = 0;ir<nrxx; ir++)
//        {
//            if(!use_libxc){
//                std::cerr << "Error: Multi-collinear approach does not support running without Libxc." << std::endl;
//                std::exit(EXIT_FAILURE);
//            }
//            double exc = 0.0;
//            for(int ipol=0;ipol<4;ipol++){
//                v(ipol, ir) = 0;
//            }
//
//            std::vector<double> n = {chr->rho[0][ir] + chr->rho_core[ir]};
//            std::vector<double> mx = {chr->rho[1][ir]};
//            std::vector<double> my = {chr->rho[2][ir]};
//            std::vector<double> mz = {chr->rho[3][ir]};
//            double amag = sqrt( pow(chr->rho[1][ir],2) + pow(chr->rho[2][ir],2) + pow(chr->rho[3][ir],2) );
//            if (n[0] - amag <= 0.0) { //ensure the rhoup and rhodn to libxc are positive
//                continue;
//            }
//            std::vector<double> E_MC;
//            std::vector<Matrix2x2> V_MC;
//            for(const int &id : func_id){
//                std::tie(E_MC, V_MC) = NCLibxc::lda_mc(id, n, mx, my, mz);
//                exc = e2*E_MC[0];
//                v(0, ir) += std::real(e2*(V_MC[0][0][0]+V_MC[0][1][1])/two);
//                v(1, ir) += std::real(e2*(V_MC[0][0][1]+V_MC[0][1][0])/two);
//                v(2, ir) += std::real(e2*(V_MC[0][1][0]-V_MC[0][0][1])/twoi);
//                v(3, ir) += std::real(e2*(V_MC[0][0][0]-V_MC[0][1][1])/two);
//                etxc += exc * n[0];
//                vtxc += v(0, ir) * chr->rho[0][ir] + v(1, ir) * chr->rho[1][ir] + v(2, ir) * chr->rho[2][ir] + v(3, ir) * chr->rho[3][ir];
//            }
//        }
    }
   

    // add gradient corrections (if any)
    // mohan modify 2009-12-15

    // the dummy variable dum contains gradient correction to stress
    // which is not used here
    std::vector<double> dum;
    if(PARAM.inp.nspin == 4 && PARAM.inp.multicolin == 1) {
        if(func_type == 2 || func_type == 3) {
            //NCLibxc::print_NCLibxc();
        }
        if(func_type == 4 || func_type == 5) {
            std::cerr << "Error: Multi-collinear approach hasn't support hybrid functioanl yet" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    } else {
        gradcorr(etxc, vtxc, v, chr, chr->rhopw, ucell, dum);
    }

    // parallel code : collect vtxc,etxc
    // mohan add 2008-06-01
#ifdef __MPI
    Parallel_Reduce::reduce_pool(etxc);
    Parallel_Reduce::reduce_pool(vtxc);
#endif
    etxc *= ucell->omega / chr->rhopw->nxyz;
    vtxc *= ucell->omega / chr->rhopw->nxyz;

    ModuleBase::timer::tick("XC_Functional","v_xc");
    return std::make_tuple(etxc, vtxc, std::move(v));
}
