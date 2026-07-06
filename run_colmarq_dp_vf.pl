#!/usr/bin/perl
use strict;
use warnings;

$| = 1;  # Autoflush output

sub run_ft2_to_json_and_binary {
    my ($input_file, $file_name_prefix, $execDir) = @_;

    # Construct the command
    my $command = $execDir . "deep-picker/release/plot_spectrum_2d "
                . "-in $input_file "
                . "-out-json ${file_name_prefix}-spectrum-information.json "
                . "-out-spectrum ${file_name_prefix}.bin > plot_${file_name_prefix}_spectrum.log 2>&1";
    # Execute command
    system($command);
    return 0;  # success
}

sub run_dp_vf_colmarm {

    my ($algorithm, $scale1, $nround, $fitting_function, $execDir, $fileDir) = @_;

    # read text file input_files.txt to get list of input_files
    # and read session.txt to get list of colmarm_user_job_ids
    
    my $input_files_file = 'input_files.txt';
    my $session_file = 'sessions.txt';
    open my $input_fh, '<', $input_files_file or die "Could not open '$input_files_file' $!";
    my @input_files = <$input_fh>;
    chomp @input_files;
    close $input_fh;
    open my $session_fh, '<', $session_file or die "Could not open '$session_file' $!";
    my @colmarm_user_job_ids = <$session_fh>;
    chomp @colmarm_user_job_ids;
    close $session_fh;

    # for all colmarm_user_job_ids, run peak picking and fitting
    # input_files have same size as colmarm_user_job_ids, one to one correspondence
    for (my $i = 0; $i < @colmarm_user_job_ids; $i++)
    {
        my $colmarm_user_job_id = $colmarm_user_job_ids[$i];
        my $input_file = $input_files[$i];
        my $colmarm_work_dir = $fileDir . "/colmarm_" . $colmarm_user_job_id;

        chdir($colmarm_work_dir) or die "Failed to change directory to $colmarm_work_dir: $!";

        unless (-e "$colmarm_work_dir/deep_peak_fit_output.txt")
        {
            unless (-e "$colmarm_work_dir/picked_peaks.json") {
                my $cmd;
                if ($algorithm eq "Simple") {
                    $cmd = $execDir . 'deep-picker/release/simple_picker -out peaks.tab picked_peaks.json -scale ' . $scale1 . ' -in ' . $input_file . ' > picking_result.txt';
                }
                elsif ($algorithm eq "DEEP1") {
                    $cmd = $execDir . 'deep-picker/release/deep_picker -out peaks.tab picked_peaks.json -scale ' . $scale1 . ' -in ' . $input_file . ' -model 1 -auto_ppp no > deep_picking_result.txt';
                }
                elsif ($algorithm eq "DEEP2") {
                    $cmd = $execDir . 'deep-picker/release/deep_picker -out peaks.tab picked_peaks.json -scale ' . $scale1 . ' -in ' . $input_file . ' -model 2 -auto_ppp no > deep_picking_result.txt';
                }
                elsif ($algorithm eq "Auto") {
                    $cmd = $execDir . 'deep-picker/release/deep_picker -out peaks.tab picked_peaks.json -scale ' . $scale1 . ' -in ' . $input_file . ' -model 2 -auto_ppp yes > deep_picking_result.txt';
                }
                print "Peak picking command executed for $colmarm_user_job_id\n";
                system($cmd);
                
            }
            else
            {
                # This is for backward compatibility only.
                if (!-e "$colmarm_work_dir/peaks.tab" && -e "$colmarm_work_dir/peaks_picked.tab") {
                    rename("$colmarm_work_dir/peaks_picked.tab", "$colmarm_work_dir/peaks.tab");
                }
            }

            my $cmd2 = $execDir . 'deep-picker/release/voigt_fit -peak_in picked_peaks.json -out fitted_peaks.json fitted.tab '
                    . ' -combine 0.0 '
                    . ' -folder ./ '
                    . ' -in ' . $input_file . ' -scale ' . $scale1 . ' -maxround ' . $nround
                    . ' -method ' . $fitting_function . ' >deep_peak_fit_output.txt 2>deep_peak_fit_erro.txt';
            print "Peak fitting command executed for $colmarm_user_job_id\n";
            system($cmd2);

           
            if (-e "$colmarm_work_dir/recon_gaussian_hsqc.ft2")
            {
                run_ft2_to_json_and_binary("recon_gaussian_hsqc.ft2", "hsqcrecon", $execDir);
            } elsif (-e "$colmarm_work_dir/recon_voigt_hsqc.ft2") {
                run_ft2_to_json_and_binary("recon_voigt_hsqc.ft2", "hsqcrecon", $execDir);
            }
        }
        else
        {
            # This is for backward compatibility only.
            if (-e "$colmarm_work_dir/peaks.tab" && !-e "$colmarm_work_dir/fitted.tab") {
                rename("$colmarm_work_dir/peaks.tab", "$colmarm_work_dir/fitted.tab");
            }
            print "Skipping peak picking and fitting for $colmarm_user_job_id as peak fitting result already exists.\n";
        }
    }
    return 0;  # success
}

sub main {
    my $argc = @ARGV;
    my ($algorithm, $scale1, $nround, $fitting_function, $execDir, $fileDir) = @ARGV;

    run_dp_vf_colmarm($algorithm, $scale1, $nround, $fitting_function, $execDir, $fileDir);

    exit 0;
}

main();
