#include "sto_forces.h"
#include "global.h"
#include "vdwd2.h"
#include "vdwd3.h"				  
#include "../module_symmetry/symmetry.h"
#include "../module_surchem/efield.h"
#include "../module_base/mathzone.h"

// new
#include "../module_xc/xc_functional.h"
#include "../module_base/math_integral.h"
#include "../src_parallel/parallel_reduce.h"
#include "../module_base/timer.h"



void Sto_Forces::init(ModuleBase::matrix& force, Stochastic_WF& stowf)
{
	ModuleBase::timer::tick("Sto_Force","cal_force");
	ModuleBase::TITLE("Sto_Forces", "init");
	this->nat =  GlobalC::ucell.nat;
	force.create(nat, 3);
	
	ModuleBase::matrix forcelc(nat, 3);
	ModuleBase::matrix forceion(nat, 3);
	ModuleBase::matrix forcecc(nat, 3);
	ModuleBase::matrix forcenl(nat, 3);
    ModuleBase::matrix sto_forcenl(nat,3);
	ModuleBase::matrix forcescc(nat, 3);
    this->cal_force_loc(forcelc, GlobalC::rhopw);
    this->cal_force_ew(forceion, GlobalC::rhopw);
    if(GlobalV::NBANDS > 0 && GlobalV::MY_STOGROUP == 0)  this->cal_force_nl(forcenl);
    this->cal_sto_force_nl(sto_forcenl,stowf);
    forcenl = forcenl + sto_forcenl;
	this->cal_force_cc(forcecc, GlobalC::rhopw);
	this->cal_force_scc(forcescc, GlobalC::rhopw);
	
    //impose total force = 0
    int iat = 0;

	ModuleBase::matrix force_e;
	if(GlobalV::EFIELD_FLAG)
	{
		force_e.create( GlobalC::ucell.nat, 3);
		Efield::compute_force(GlobalC::ucell, force_e);
	}
	
	for (int ipol = 0; ipol < 3; ipol++)
	{
		double sum = 0.0;
		iat = 0;

		for (int it = 0;it <  GlobalC::ucell.ntype;it++)
		{
			for (int ia = 0;ia <  GlobalC::ucell.atoms[it].na;ia++)
			{
				force(iat, ipol) =
					forcelc(iat, ipol)
					+ forceion(iat, ipol)
					+ forcenl(iat, ipol)
					+ forcecc(iat, ipol)
					+ forcescc(iat, ipol);																									   
					
				if(GlobalV::EFIELD_FLAG)
				{
					force(iat,ipol) = force(iat, ipol) + force_e(iat, ipol);
				}

				sum += force(iat, ipol);

				iat++;
			}
		}

		double compen = sum /  GlobalC::ucell.nat;
		for(int iat=0; iat<GlobalC::ucell.nat; ++iat)
		{
			force(iat, ipol) = force(iat, ipol) - compen;
		}	
	}
	
	if(ModuleSymmetry::Symmetry::symm_flag)
	{
		double *pos;
		double d1,d2,d3;
		pos = new double[GlobalC::ucell.nat*3];
		ModuleBase::GlobalFunc::ZEROS(pos, GlobalC::ucell.nat*3);
		int iat = 0;
		for(int it = 0;it < GlobalC::ucell.ntype;it++)
		{
			//Atom* atom = &ucell.atoms[it];
			for(int ia =0;ia< GlobalC::ucell.atoms[it].na;ia++)
			{
				pos[3*iat  ] = GlobalC::ucell.atoms[it].taud[ia].x ;
				pos[3*iat+1] = GlobalC::ucell.atoms[it].taud[ia].y ;
				pos[3*iat+2] = GlobalC::ucell.atoms[it].taud[ia].z;
				for(int k=0; k<3; ++k)
				{
					GlobalC::symm.check_translation( pos[iat*3+k], -floor(pos[iat*3+k]));
					GlobalC::symm.check_boundary( pos[iat*3+k] );
				}
				iat++;				
			}
		}
		
		for(int iat=0; iat<GlobalC::ucell.nat; iat++)
		{
			ModuleBase::Mathzone::Cartesian_to_Direct(force(iat,0),force(iat,1),force(iat,2),
                                        GlobalC::ucell.a1.x, GlobalC::ucell.a1.y, GlobalC::ucell.a1.z,
                                        GlobalC::ucell.a2.x, GlobalC::ucell.a2.y, GlobalC::ucell.a2.z,
                                        GlobalC::ucell.a3.x, GlobalC::ucell.a3.y, GlobalC::ucell.a3.z,
                                        d1,d2,d3);
			
			force(iat,0) = d1;force(iat,1) = d2;force(iat,2) = d3;
		}
		GlobalC::symm.force_symmetry(force , pos, GlobalC::ucell);
		for(int iat=0; iat<GlobalC::ucell.nat; iat++)
		{
			ModuleBase::Mathzone::Direct_to_Cartesian(force(iat,0),force(iat,1),force(iat,2),
                                        GlobalC::ucell.a1.x, GlobalC::ucell.a1.y, GlobalC::ucell.a1.z,
                                        GlobalC::ucell.a2.x, GlobalC::ucell.a2.y, GlobalC::ucell.a2.z,
                                        GlobalC::ucell.a3.x, GlobalC::ucell.a3.y, GlobalC::ucell.a3.z,
                                        d1,d2,d3);
			force(iat,0) = d1;force(iat,1) = d2;force(iat,2) = d3;
		}
		// cout << "nrotk =" << symm.nrotk << endl;
		delete[] pos;
		
	}

 	GlobalV::ofs_running << setiosflags(ios::fixed) << setprecision(6) << endl;
	if(GlobalV::TEST_FORCE)
	{
		Sto_Forces::print("LOCAL    FORCE (Ry/Bohr)", forcelc);
		Sto_Forces::print("NONLOCAL FORCE (Ry/Bohr)", forcenl);
		Sto_Forces::print("NLCC     FORCE (Ry/Bohr)", forcecc);
		Sto_Forces::print("ION      FORCE (Ry/Bohr)", forceion);
		Sto_Forces::print("SCC      FORCE (Ry/Bohr)", forcescc);
		if(GlobalV::EFIELD_FLAG) Sto_Forces::print("EFIELD   FORCE (Ry/Bohr)", force_e);
	}
	
		
	// output force in unit eV/Angstrom
	GlobalV::ofs_running << endl;
    
	if(GlobalV::TEST_FORCE)
	{
		Sto_Forces::print("LOCAL    FORCE (eV/Angstrom)", forcelc,0);
		Sto_Forces::print("NONLOCAL FORCE (eV/Angstrom)", forcenl,0);
		Sto_Forces::print("NLCC     FORCE (eV/Angstrom)", forcecc,0);
		Sto_Forces::print("ION      FORCE (eV/Angstrom)", forceion,0);
		Sto_Forces::print("SCC      FORCE (eV/Angstrom)", forcescc,0);
		if(GlobalV::EFIELD_FLAG) Sto_Forces::print("EFIELD   FORCE (eV/Angstrom)", force_e,0);
	}
	Sto_Forces::print("   TOTAL-FORCE (eV/Angstrom)", force,0);
	ModuleBase::timer::tick("Sto_Force","cal_force");
    return;
}

void Sto_Forces::cal_force_nl(ModuleBase::matrix& forcenl)
{
	ModuleBase::TITLE("Sto_Forces","cal_force_nl");
	ModuleBase::timer::tick("Sto_Forces","cal_force_nl");

    const int nkb = GlobalC::ppcell.nkb;
	if(nkb == 0) return; // mohan add 2010-07-25
	
	// dbecp: conj( -iG * <Beta(nkb,npw)|psi(nbnd,npw)> )
	ModuleBase::ComplexArray dbecp( nkb, GlobalV::NBANDS, 3);
    ModuleBase::ComplexMatrix becp( nkb, GlobalV::NBANDS);
    
	
	// vkb1: |Beta(nkb,npw)><Beta(nkb,npw)|psi(nbnd,npw)>
	ModuleBase::ComplexMatrix vkb1( nkb, GlobalC::wf.npwx );

    for (int ik = 0;ik < GlobalC::kv.nks;ik++)
    {
        if (GlobalV::NSPIN==2) GlobalV::CURRENT_SPIN = GlobalC::kv.isk[ik];
        GlobalC::wf.npw = GlobalC::kv.ngk[ik];
        // generate vkb
        if (GlobalC::ppcell.nkb > 0)
        {
            GlobalC::ppcell.getvnl(ik);
        }

        // get becp according to wave functions and vkb
        // important here ! becp must set zero!!
		// vkb: Beta(nkb,npw)
		// becp(nkb,nbnd): <Beta(nkb,npw)|psi(nbnd,npw)>
        becp.zero_out();
        for (int ib=0; ib<GlobalV::NBANDS; ib++)
        {
            for (int i=0;i<nkb;i++)
            {
                for (int ig=0; ig<GlobalC::wf.npw; ig++)
                {
                    becp(i,ib) += GlobalC::wf.evc[ik](ib,ig)* conj( GlobalC::ppcell.vkb(i,ig) );
                }
            }
        }
        Parallel_Reduce::reduce_complex_double_pool( becp.c, becp.size);

        //out.printcm_real("becp",becp,1.0e-4);
        // Calculate the derivative of beta,
        // |dbeta> =  -ig * |beta>
        dbecp.zero_out();
        for (int ipol = 0; ipol<3; ipol++)
        {
			for (int i = 0;i < nkb;i++)
			{
				if (ipol==0)
				{
					for (int ig=0; ig<GlobalC::wf.npw; ig++)
						vkb1(i, ig) = GlobalC::ppcell.vkb(i, ig) * ModuleBase::NEG_IMAG_UNIT * GlobalC::pw.gcar[ GlobalC::wf.igk(ik, ig) ].x;
				}
				if (ipol==1)
				{
					for (int ig=0; ig<GlobalC::wf.npw; ig++)
						vkb1(i, ig) = GlobalC::ppcell.vkb(i, ig) * ModuleBase::NEG_IMAG_UNIT * GlobalC::pw.gcar[ GlobalC::wf.igk(ik, ig) ].y;
				}
				if (ipol==2)
				{
					for (int ig=0; ig<GlobalC::wf.npw; ig++)
						vkb1(i, ig) = GlobalC::ppcell.vkb(i, ig) * ModuleBase::NEG_IMAG_UNIT * GlobalC::pw.gcar[ GlobalC::wf.igk(ik, ig) ].z;
				}
			}
            for (int ib=0; ib<GlobalV::NBANDS; ib++)
            {
                for (int i=0; i<nkb; i++)
                {
                    for (int ig=0; ig<GlobalC::wf.npw; ig++)
                    {
                        dbecp(i,ib, ipol) += conj( vkb1(i,ig) ) * GlobalC::wf.evc[ik](ib,ig) ;
                    }
                }
            }
        }// end ipol

//		don't need to reduce here, keep dbecp different in each processor,
//		and at last sum up all the forces.
//		Parallel_Reduce::reduce_complex_double_pool( dbecp.ptr, dbecp.ndata);

//		double *cf = new double[ucell.nat*3];
//		ZEROS(cf, ucell.nat);
		for (int ib=0; ib<GlobalV::NBANDS; ib++)
		{
			double fac = GlobalC::wf.wg(ik, ib) * 2.0 * GlobalC::ucell.tpiba;
        	int iat = 0;
        	int sum = 0;
			for (int it=0; it<GlobalC::ucell.ntype; it++)
			{
				const int Nprojs = GlobalC::ucell.atoms[it].nh;
				for (int ia=0; ia<GlobalC::ucell.atoms[it].na; ia++)
				{
					for (int ip=0; ip<Nprojs; ip++)
					{
						double ps = GlobalC::ppcell.deeq(GlobalV::CURRENT_SPIN, iat, ip, ip) ;
						const int inkb = sum + ip; 
						//out<<"\n ps = "<<ps;

						for (int ipol=0; ipol<3; ipol++)
						{
							const double dbb = ( conj( dbecp( inkb, ib, ipol) ) * becp( inkb, ib) ).real();
							forcenl(iat, ipol) = forcenl(iat, ipol) - ps * fac * dbb;
							//cf[iat*3+ipol] += ps * fac * dbb;
						}
					}

					//if ( ucell.atoms[it].nbeta > ucell.atoms[it].lmax+1 )    //{zws add 20160110
					//{
					//cout << " \n multi-projector force calculation ... " << endl;
					for (int ip=0; ip<Nprojs; ip++)
					{
						const int inkb = sum + ip;
						//for (int ip2=0; ip2<Nprojs; ip2++)
						for (int ip2=ip+1; ip2<Nprojs; ip2++)
						{
						//if ( ip != ip2 )
						//{
							const int jnkb = sum + ip2;
							double ps = GlobalC::ppcell.deeq(GlobalV::CURRENT_SPIN, iat, ip2, ip) ;

							for (int ipol=0; ipol<3; ipol++)
							{
								const double dbb = ( conj( dbecp( inkb, ib, ipol) ) * becp( jnkb, ib)
										+ dbecp( jnkb, ib, ipol) * conj(becp( inkb, ib) ) ).real();
								//const double dbb = ( conj( dbecp( inkb, ib, ipol) ) * becp( jnkb, ib) ).real();
								forcenl(iat, ipol) = forcenl(iat, ipol) - ps * fac * dbb;
								//cf[iat*3+ipol] += ps * fac * dbb;
							}
						//}
						}
					}
					//}    //}zws add 20160110

					++iat;
					sum+=Nprojs;
				}
			} //end it
		} //end band
    }// end ik

    // sum up forcenl from all processors
    Parallel_Reduce::reduce_double_pool(forcenl.c, forcenl.nr * forcenl.nc);
//  this->print(ofs_running, "nonlocal forces", forcenl);
	ModuleBase::timer::tick("Sto_Forces","cal_force_nl");
    return;
}

void Sto_Forces::cal_sto_force_nl(ModuleBase::matrix& forcenl, Stochastic_WF& stowf)
{
	ModuleBase::TITLE("Sto_Forces","cal_force_nl");
	ModuleBase::timer::tick("Sto_Forces","cal_force_nl");

    const int nkb = GlobalC::ppcell.nkb;
    int* nchip = stowf.nchip;
	if(nkb == 0) return; // mohan add 2010-07-25
	
	// dbecp: conj( -iG * <Beta(nkb,npw)|psi(nbnd,npw)> )
    
	
	// vkb1: |Beta(nkb,npw)><Beta(nkb,npw)|psi(nbnd,npw)>
	ModuleBase::ComplexMatrix vkb1( nkb, GlobalC::wf.npwx );

    for (int ik = 0;ik < GlobalC::kv.nks;ik++)
    {
		ModuleBase::ComplexArray dbecp( nkb, nchip[ik], 3);
    	ModuleBase::ComplexMatrix becp( nkb, nchip[ik]);
        if (GlobalV::NSPIN==2) GlobalV::CURRENT_SPIN = GlobalC::kv.isk[ik];
        GlobalC::wf.npw = GlobalC::kv.ngk[ik];
        // generate vkb
        if (GlobalC::ppcell.nkb > 0)
        {
            GlobalC::ppcell.getvnl(ik);
        }

        // get becp according to wave functions and vkb
        // important here ! becp must set zero!!
		// vkb: Beta(nkb,npw)
		// becp(nkb,nbnd): <Beta(nkb,npw)|psi(nbnd,npw)>
        becp.zero_out();
        for (int ib=0; ib<nchip[ik]; ib++)
        {
            for (int i=0;i<nkb;i++)
            {
                for (int ig=0; ig<GlobalC::wf.npw; ig++)
                {
                    becp(i,ib) += stowf.shchi[ik](ib,ig)* conj( GlobalC::ppcell.vkb(i,ig) );
                }
            }
        }
        Parallel_Reduce::reduce_complex_double_pool( becp.c, becp.size);

        //out.printcm_real("becp",becp,1.0e-4);
        // Calculate the derivative of beta,
        // |dbeta> =  -ig * |beta>
        dbecp.zero_out();
        for (int ipol = 0; ipol<3; ipol++)
        {
			for (int i = 0;i < nkb;i++)
			{
				if (ipol==0)
				{
					for (int ig=0; ig<GlobalC::wf.npw; ig++)
						vkb1(i, ig) = GlobalC::ppcell.vkb(i, ig) * ModuleBase::NEG_IMAG_UNIT * GlobalC::pw.gcar[ GlobalC::wf.igk(ik, ig) ].x;
				}
				if (ipol==1)
				{
					for (int ig=0; ig<GlobalC::wf.npw; ig++)
						vkb1(i, ig) = GlobalC::ppcell.vkb(i, ig) * ModuleBase::NEG_IMAG_UNIT * GlobalC::pw.gcar[ GlobalC::wf.igk(ik, ig) ].y;
				}
				if (ipol==2)
				{
					for (int ig=0; ig<GlobalC::wf.npw; ig++)
						vkb1(i, ig) = GlobalC::ppcell.vkb(i, ig) * ModuleBase::NEG_IMAG_UNIT * GlobalC::pw.gcar[ GlobalC::wf.igk(ik, ig) ].z;
				}
			}
            for (int ib=0; ib<nchip[ik]; ib++)
            {
                for (int i=0; i<nkb; i++)
                {
                    for (int ig=0; ig<GlobalC::wf.npw; ig++)
                    {
                        dbecp(i,ib, ipol) += conj( vkb1(i,ig) ) * stowf.shchi[ik](ib,ig) ;
                    }
                }
            }
        }// end ipol

//		don't need to reduce here, keep dbecp different in each processor,
//		and at last sum up all the forces.
//		Parallel_Reduce::reduce_complex_double_pool( dbecp.ptr, dbecp.ndata);

//		double *cf = new double[ucell.nat*3];
//		ZEROS(cf, ucell.nat);
		for (int ib=0; ib<nchip[ik]; ib++)
		{
			double fac = GlobalC::kv.wk[ik] * 2.0 * GlobalC::ucell.tpiba;
        	int iat = 0;
        	int sum = 0;
			for (int it=0; it< GlobalC::ucell.ntype; it++)
			{
				const int Nprojs =  GlobalC::ucell.atoms[it].nh;
				for (int ia=0; ia< GlobalC::ucell.atoms[it].na; ia++)
				{
					for (int ip=0; ip<Nprojs; ip++)
					{
						double ps =  GlobalC::ppcell.deeq( GlobalV::CURRENT_SPIN, iat, ip, ip) ;
						const int inkb = sum + ip; 
						//out<<"\n ps = "<<ps;

						for (int ipol=0; ipol<3; ipol++)
						{
							const double dbb = ( conj( dbecp( inkb, ib, ipol) ) * becp( inkb, ib) ).real();
							forcenl(iat, ipol) = forcenl(iat, ipol) - ps * fac * dbb;
							//cf[iat*3+ipol] += ps * fac * dbb;
						}
					}

					//if ( ucell.atoms[it].nbeta > ucell.atoms[it].lmax+1 )    //{zws add 20160110
					//{
					//cout << " \n multi-projector force calculation ... " << endl;
					for (int ip=0; ip<Nprojs; ip++)
					{
						const int inkb = sum + ip;
						//for (int ip2=0; ip2<Nprojs; ip2++)
						for (int ip2=ip+1; ip2<Nprojs; ip2++)
						{
						//if ( ip != ip2 )
						//{
							const int jnkb = sum + ip2;
							double ps =  GlobalC::ppcell.deeq( GlobalV::CURRENT_SPIN, iat, ip2, ip) ;

							for (int ipol=0; ipol<3; ipol++)
							{
								const double dbb = ( conj( dbecp( inkb, ib, ipol) ) * becp( jnkb, ib)
										+ dbecp( jnkb, ib, ipol) * conj(becp( inkb, ib) ) ).real();
								//const double dbb = ( conj( dbecp( inkb, ib, ipol) ) * becp( jnkb, ib) ).real();
								forcenl(iat, ipol) = forcenl(iat, ipol) - ps * fac * dbb;
								//cf[iat*3+ipol] += ps * fac * dbb;
							}
						//}
						}
					}
					//}    //}zws add 20160110

					++iat;
					sum+=Nprojs;
				}
			} //end it
		} //end band
    }// end ik

    // sum up forcenl from all processors
    Parallel_Reduce::reduce_double_all(forcenl.c, forcenl.nr * forcenl.nc);
//  this->print(ofs_running, "nonlocal forces", forcenl);
	ModuleBase::timer::tick("Sto_Forces","cal_force_nl");
    return;
}


